"""
Intake agent for ticket classification.

This is Level 0 in the orchestration hierarchy. Runs on EVERY ticket.
Uses fast LLM for cost efficiency.
"""

import hashlib
import json
import logging
from typing import Any

from langfuse import observe

from config import settings
from core.agents.state import (
    ComplexityRoute,
    PrimaryTopic,
    TicketState,
    UrgencyLevel,
)
from core.llm_client import call_fast
from core.redis_client import get_redis_client

from abc import ABC, abstractmethod
import re

logger = logging.getLogger(__name__)

class BaseClassifier(ABC):
    """Abstract base class for all intake classification backends."""

    @abstractmethod
    async def classify(
        self, ticket_content: str, chat_history: list[dict] | None = None
    ) -> dict[str, Any] | None:
        """
        Classify ticket content.

        Args:
            ticket_content: The support ticket body string.
            chat_history: Optional previous chat history turns.

        Returns:
            Dictionary containing complexity, urgency, primary_topic,
            error_codes, search_keywords, and confidence, or None on failure.
        """
        pass


class LLMClassifier(BaseClassifier):
    """Prompt-based LLM classifier using fast-LLM model calls."""

    def __init__(self, agent: "IntakeAgent") -> None:
        self.agent = agent

    async def classify(
        self, ticket_content: str, chat_history: list[dict] | None = None
    ) -> dict[str, Any] | None:
        # First attempt
        result = await self.agent._call_fast_llm(ticket_content, is_retry=False, chat_history=chat_history)
        
        if result is None:
            # Retry with explicit JSON instruction
            logger.warning("First parse failed, retrying with explicit prompt")
            result = await self.agent._call_fast_llm(ticket_content, is_retry=True, chat_history=chat_history)
            
        return result


class VerifiedLLMClassifier(BaseClassifier):
    """
    Runs LLMClassifier, then cross-checks the result against
    DeterministicHeuristicClassifier. Flags disagreements as
    low-confidence for adaptive routing.
    """

    def __init__(self, agent: "IntakeAgent") -> None:
        self._llm = LLMClassifier(agent)
        self._heuristic = DeterministicHeuristicClassifier()

    async def classify(
        self, ticket_content: str, chat_history: list[dict] | None = None
    ) -> dict[str, Any] | None:
        llm_result = await self._llm.classify(ticket_content, chat_history=chat_history)
        heuristic_result = await self._heuristic.classify(ticket_content, chat_history=chat_history)

        if llm_result and heuristic_result:
            # If urgency disagrees, flag as lower confidence
            if llm_result.get("urgency") != heuristic_result.get("urgency"):
                original_confidence = llm_result.get("confidence", 0.85)
                llm_result["confidence"] = min(original_confidence, 0.55)
                llm_result["classification_disagreement"] = True
                logger.info(
                    f"Heuristic cross-check: LLM urgency '{llm_result.get('urgency')}' "
                    f"disagrees with Heuristic urgency '{heuristic_result.get('urgency')}'. "
                    f"Downgrading confidence from {original_confidence} to 0.55."
                )

            # If complexity disagrees dramatically (simple vs complex), downgrade confidence too
            llm_complexity = llm_result.get("complexity", "moderate")
            heur_complexity = heuristic_result.get("complexity", "moderate")
            if (llm_complexity == "simple" and heur_complexity == "complex") or \
               (llm_complexity == "complex" and heur_complexity == "simple"):
                original_confidence = llm_result.get("confidence", 0.85)
                llm_result["confidence"] = min(original_confidence, 0.55)
                llm_result["classification_disagreement"] = True
                logger.info(
                    f"Heuristic cross-check: LLM complexity '{llm_complexity}' "
                    f"disagrees with Heuristic complexity '{heur_complexity}'. "
                    f"Downgrading confidence from {original_confidence} to 0.55."
                )

        return llm_result


class DeterministicHeuristicClassifier(BaseClassifier):
    """
    A zero-dependency local heuristic classifier.

    Uses keyword mapping and regex rules to classify ticket categories,
    complexity, and urgency without incurring LLM or network latency.
    Useful for offline testing, local dev, or fallback systems.
    """

    def __init__(self) -> None:
        # Simple regex rules for topics
        self.topic_rules = {
            "webhook": re.compile(r"webhook|signature|signing|endpoint", re.I),
            "billing": re.compile(r"billing|invoice|payment|charge|refund|card|subscription", re.I),
            "connect": re.compile(r"connect|express|customaccount|capabilities|platform", re.I),
            "auth": re.compile(r"auth|login|token|api_key|secret|credential|permission", re.I),
            "radar": re.compile(r"radar|fraud|dispute|block|chargeback", re.I),
            "api": re.compile(r"api|request|endpoint|curl|sdk|method", re.I),
        }
        # Common error code patterns
        self.error_code_pattern = re.compile(r"\b([a-z0-9_]+_error|[a-z0-9_]+_missing|err_[a-z0-9_]+)\b", re.I)

    async def classify(
        self, ticket_content: str, chat_history: list[dict] | None = None
    ) -> dict[str, Any] | None:
        content_lower = ticket_content.lower()
        
        # 1. Determine primary topic
        primary_topic = "other"
        for topic, pattern in self.topic_rules.items():
            if pattern.search(ticket_content):
                primary_topic = topic
                break
                
        # 2. Extract error codes
        error_codes = list(set(self.error_code_pattern.findall(ticket_content)))
        
        # 3. Determine urgency
        urgency = "medium"
        if any(w in content_lower for w in ["production down", "prod down", "urgent", "critical", "suspension", "blocker"]):
            urgency = "high"
        elif any(w in content_lower for w in ["data loss", "revenue", "loss", "sla breach"]):
            urgency = "critical"
        elif any(w in content_lower for w in ["just wondering", "general question", "minor", "low"]):
            urgency = "low"
            
        # 4. Determine complexity
        complexity = "moderate"
        if len(ticket_content) < 120 and len(error_codes) == 0:
            complexity = "simple"
        elif len(ticket_content) > 400 or len(error_codes) > 1 or "connect" in content_lower:
            complexity = "complex"
            
        # 5. Search keywords (simple extract)
        words = [w for w in re.findall(r"\b[A-Za-z]{4,15}\b", ticket_content) if w.lower() not in ["with", "from", "that", "this", "your", "have"]]
        search_keywords = list(set(words[:5]))
        
        return {
            "complexity": complexity,
            "urgency": urgency,
            "primary_topic": primary_topic,
            "error_codes": error_codes,
            "search_keywords": search_keywords,
            "confidence": 0.70,
        }
class BaseRouter(ABC):
    """Abstract base class for all ticket routers."""

    @abstractmethod
    def route(self, state: TicketState) -> str:
        """
        Determine routing path.

        Args:
            state: Full TicketState.

        Returns:
            Route name: 'simple_retrieval', 'parallel_retrieval', or 'escalate'.
        """
        pass


class StaticRuleRouter(BaseRouter):
    """Encapsulates the standard rule-based static routing."""

    def route(self, state: TicketState) -> str:
        complexity = state.get("complexity", "moderate")
        urgency = state.get("urgency", "medium")
        ticket_content = state.get("ticket_content", "").lower()

        # Immediate escalation for critical + data loss
        if urgency == "critical" and "data_loss" in ticket_content:
            logger.info("Routing to escalate: critical urgency with data_loss")
            return "escalate"

        # Also escalate for critical + production failures
        if urgency == "critical" and any(
            term in ticket_content
            for term in ["production", "prod", "all payments", "completely down"]
        ):
            logger.info("Routing to escalate: critical production issue")
            return "escalate"

        # Simple tickets get focused retrieval
        if complexity == "simple":
            logger.info("Routing to simple_retrieval")
            return "simple_retrieval"

        # Moderate and complex both get parallel retrieval
        logger.info("Routing to parallel_retrieval")
        return "parallel_retrieval"


class ConfidenceAdaptiveRouter(BaseRouter):
    """
    An adaptive routing strategy that adjusts the routing decision
    based on customer tier, classification confidence, and topic sensitivity,
    incorporating a downstream feedback loop (sufficiency, grounding, and escalation).
    """

    def route(self, state: TicketState) -> str:
        complexity = state.get("complexity", "moderate")
        urgency = state.get("urgency", "medium")
        confidence = state.get("intake_confidence", 1.0)
        customer_tier = state.get("customer_tier", "standard")
        topic = state.get("primary_topic", "other")
        ticket_content = state.get("ticket_content", "").lower()

        # Rule 0: Downstream Feedback Loop (learning from historical sufficiency and grounding quality)
        feedback_history = state.get("feedback_history")
        if feedback_history:
            prev_route = feedback_history.get("route_taken", "")
            quality_score = feedback_history.get("quality_score", 1.0)
            synthesis_confidence = feedback_history.get("synthesis_confidence", 1.0)
            gaps = feedback_history.get("knowledge_gaps", [])
            issues = feedback_history.get("quality_issues", [])
            escalated = feedback_history.get("escalated", False)

            # Define failure based on low grounding quality or low retrieval/sufficiency confidence
            has_grounding_failure = quality_score < 0.60 or len(issues) > 0
            has_sufficiency_failure = synthesis_confidence < 0.50 or len(gaps) > 0

            if has_grounding_failure or has_sufficiency_failure or escalated:
                logger.info(
                    f"[FEEDBACK LOOP DETECTED] Grounding/sufficiency failure in previous run! "
                    f"Prev Route={prev_route}, Quality Score={quality_score:.2f}, "
                    f"Synthesis Confidence={synthesis_confidence:.2f}, Escalated={escalated}"
                )

                # Feedback Loop Routing Upgrade 1: Simple retrieval previously failed
                # Upgrade to parallel retrieval to pull broader changes & community Q&A
                if prev_route in ("simple", "simple_retrieval"):
                    logger.info("[FEEDBACK LOOP] Upgrading route simple -> parallel_retrieval due to downstream grounding/sufficiency failure")
                    return "parallel_retrieval"

                # Feedback Loop Routing Upgrade 2: Parallel retrieval also failed
                # Immediately escalate to human support to avoid infinite loops and wasting LLM tokens
                if prev_route in ("moderate", "complex", "parallel_retrieval"):
                    logger.info("[FEEDBACK LOOP] Upgrading route parallel_retrieval -> escalate due to recurrent downstream grounding/sufficiency failures")
                    return "escalate"

        # Rule 1: High-priority critical escalations (same as static rules)
        if urgency == "critical" and ("data_loss" in ticket_content or any(
            term in ticket_content
            for term in ["production", "prod", "all payments", "completely down"]
        )):
            logger.info("Adaptive Route -> escalate: critical production impact")
            return "escalate"

        # Rule 2: Low-confidence safety fallback
        if confidence < 0.60:
            if customer_tier == "enterprise" or urgency in ("high", "critical"):
                logger.info(f"Adaptive Route -> escalate: low classification confidence ({confidence:.2f}) on urgent/enterprise ticket")
                return "escalate"
            logger.info(f"Adaptive Route -> parallel_retrieval: low confidence ({confidence:.2f}), using broad search as safety fallback")
            return "parallel_retrieval"

        # Rule 3: High-touch escalation for sensitive regulatory topics (billing, auth) on Enterprise tiers
        if customer_tier == "enterprise" and topic in ("billing", "auth") and complexity in ("moderate", "complex"):
            logger.info("Adaptive Route -> escalate: sensitive enterprise billing/auth query")
            return "escalate"

        # Rule 4: Normal paths
        if complexity == "simple":
            # Extra check: even if simple, if it's critical or enterprise billing/auth, escalate or parallel retrieve
            if customer_tier == "enterprise" and topic in ("billing", "auth"):
                logger.info("Adaptive Route -> parallel_retrieval: enterprise simple billing/auth gets broad search")
                return "parallel_retrieval"
            logger.info("Adaptive Route -> simple_retrieval")
            return "simple_retrieval"

        logger.info("Adaptive Route -> parallel_retrieval")
        return "parallel_retrieval"


class IntakeAgent:
    """
    Classifies incoming support tickets for routing.
    
    Determines ticket complexity, urgency, primary topic, and extracts
    any error codes. Uses fast LLM for cost-efficient classification.
    """

    INTAKE_PROMPT = """Analyze this support ticket. Return JSON only.
No explanation. No markdown. Just the JSON object.

Ticket: {ticket_content}

{{
    "complexity": "simple|moderate|complex",
    "urgency": "low|medium|high|critical",
    "primary_topic": "webhook|billing|connect|auth|api|radar|other",
    "error_codes": ["list of any error codes mentioned, empty list if none"],
    "search_keywords": ["3-5 core search keywords (e.g. 'PaymentIntents', 'upgrade', 'API')"],
    "is_about_deprecated_feature": false,
    "confidence": 0.85
}}

Complexity guide:
- simple = single concept, likely answered in one doc
- moderate = requires 2-3 sources to answer
- complex = multi-system issue, edge case, no obvious answer

Urgency guide:
- low = general question, no time pressure
- medium = normal support request
- high = customer blocked, needs quick resolution
- critical = ONLY if: data loss, payment failure in production, or account suspension

Return valid JSON only."""

    CONVERSATIONAL_INTAKE_PROMPT = """Analyze this support ticket in the context of the conversation history. Return JSON only.
No explanation. No markdown. Just the JSON object.

CONVERSATION HISTORY:
{history_formatted}

NEW CUSTOMER TICKET:
{ticket_content}

{{
    "complexity": "simple|moderate|complex",
    "urgency": "low|medium|high|critical",
    "primary_topic": "webhook|billing|connect|auth|api|radar|other",
    "error_codes": ["list of any error codes mentioned, empty list if none"],
    "search_keywords": ["3-5 core search keywords (e.g. 'PaymentIntents', 'upgrade', 'API')"],
    "is_about_deprecated_feature": false,
    "confidence": 0.85,
    "topic_shift": false,
    "rewritten_query": "standalone search query"
}}

CRITICAL GUIDELINES:
1. 'topic_shift' (Boolean):
   - You MUST set this to true if the customer's new ticket shifts to a completely different, unrelated topic from the conversation history (e.g., shifting from python webhook verification to billing invoices, or from custom accounts to billing).
   - Set this to false if the ticket is a follow-up, clarification, or directly relates to the context/issues discussed in the conversation history.

2. 'rewritten_query' (String):
   - If 'topic_shift' is true, set 'rewritten_query' to the exact raw NEW CUSTOMER TICKET content.
   - If 'topic_shift' is false, rewrite the query as a standalone search query that incorporates essential context from the conversation history (e.g., specific SDKs, technical features, error codes) so that search engines can locate relevant information. Remove any ambiguous pronouns.

Return valid JSON only."""

    RETRY_PROMPT = """Your previous response was not valid JSON. 
Return ONLY a valid JSON object with no additional text.
No markdown code blocks. No explanation.

Ticket: {ticket_content}

Required JSON structure:
{{
    "complexity": "simple",
    "urgency": "medium", 
    "primary_topic": "other",
    "error_codes": [],
    "search_keywords": [],
    "is_about_deprecated_feature": false,
    "confidence": 0.5
}}"""

    def __init__(
        self,
        classifier: BaseClassifier | None = None,
        router: BaseRouter | None = None,
    ) -> None:
        """Initialize intake agent with pluggable classifier and router."""
        self.classifier = classifier or VerifiedLLMClassifier(self)
        self.router = router or StaticRuleRouter()
        logger.info(
            f"IntakeAgent initialized: classifier={self.classifier.__class__.__name__}, "
            f"router={self.router.__class__.__name__}"
        )


    @observe(name="intake_agent")
    async def process(self, state: TicketState) -> TicketState:
        """
        Classify ticket complexity, urgency, and topic.
        
        Uses fast LLM for cost-efficient classification.
        Extracts error codes if present in the ticket content.
        
        Args:
            state: Current ticket state with ticket_content.
            
        Returns:
            Updated state with intake classification outputs.
        """
        ticket_content = state.get("ticket_content", "")
        
        if not ticket_content:
            logger.warning("Empty ticket content, using defaults")
            return self._apply_defaults(state)

        chat_history = state.get("chat_history", [])
        is_conversational = len(chat_history) > 0

        # Check cache for repeated tickets to optimize tokens and latency
        clean_content = ticket_content.strip()
        content_hash = hashlib.md5(clean_content.lower().encode('utf-8')).hexdigest()
        cache_key = f"cache:intake:{content_hash}"
        
        # Load downstream feedback history for the routing feedback loop
        try:
            redis_client = get_redis_client()
            feedback_history = await redis_client.get_json(f"feedback:history:{content_hash}")
            if feedback_history:
                state["feedback_history"] = feedback_history
                logger.info(f"[FEEDBACK LOOP] Loaded downstream performance feedback for hash {content_hash}")
        except Exception as e:
            logger.warning(f"Failed to load feedback history from cache: {e}")
            
        if not is_conversational:
            try:
                redis_client = get_redis_client()
                cached = await redis_client.get_json(cache_key)
                if cached:
                    logger.info(f"[CACHE HIT] Found intake classification for query hash {content_hash}")
                    state["complexity"] = self._validate_complexity(cached.get("complexity"))
                    state["urgency"] = self._validate_urgency(cached.get("urgency"))
                    state["primary_topic"] = self._validate_topic(cached.get("primary_topic"))
                    state["error_codes"] = self._validate_error_codes(cached.get("error_codes"))
                    state["search_keywords"] = cached.get("search_keywords", [])
                    state["intake_confidence"] = self._validate_confidence(cached.get("confidence"))
                    state["topic_shift"] = False
                    state["rewritten_query"] = ticket_content
                    state["agent_path"] = ["intake"]
                    return state
            except Exception as e:
                logger.warning(f"Failed to check intake cache: {e}")
                
        # Cache miss or conversational follow-up - perform classification via pluggable classifier
        result = await self.classifier.classify(ticket_content, chat_history=chat_history)
        
        if result is None:
            # Classification failed - use safe defaults
            logger.error("Classification failed, using defaults")
            return self._apply_defaults(state)

        # Update state with parsed results
        state["complexity"] = self._validate_complexity(result.get("complexity"))
        state["urgency"] = self._validate_urgency(result.get("urgency"))
        state["primary_topic"] = self._validate_topic(result.get("primary_topic"))
        state["error_codes"] = self._validate_error_codes(result.get("error_codes"))
        state["search_keywords"] = result.get("search_keywords", [])
        state["intake_confidence"] = self._validate_confidence(result.get("confidence"))
        state["topic_shift"] = bool(result.get("topic_shift", False))
        state["rewritten_query"] = str(result.get("rewritten_query", ticket_content))
        
        # Track agent path
        state["agent_path"] = ["intake"]
        
        logger.info(
            f"Intake complete: complexity={state['complexity']}, "
            f"urgency={state['urgency']}, topic={state['primary_topic']}, "
            f"confidence={state['intake_confidence']:.2f}, "
            f"topic_shift={state['topic_shift']}, rewritten_query='{state['rewritten_query']}'"
        )
        
        # Store valid classification in cache (TTL = 24 hours) - only if not conversational
        if not is_conversational:
            try:
                redis_client = get_redis_client()
                cache_data = {
                    "complexity": state["complexity"],
                    "urgency": state["urgency"],
                    "primary_topic": state["primary_topic"],
                    "error_codes": state["error_codes"],
                    "search_keywords": state["search_keywords"],
                    "confidence": state["intake_confidence"],
                }
                await redis_client.set_json(cache_key, cache_data, ttl_seconds=86400)
                logger.info(f"[CACHE SET] Cached intake classification for query hash {content_hash}")
            except Exception as e:
                logger.warning(f"Failed to cache intake result: {e}")
            
        return state

    async def _call_fast_llm(
        self,
        ticket_content: str,
        is_retry: bool,
        chat_history: list[dict] | None = None,
    ) -> dict[str, Any] | None:
        """
        Call fast LLM (Grok) and parse JSON response.
        
        Args:
            ticket_content: The ticket text to classify.
            is_retry: Whether this is a retry attempt.
            chat_history: Optional list of previous chat history turns.
            
        Returns:
            Parsed JSON dict or None if parsing failed.
        """
        if is_retry:
            prompt = self.RETRY_PROMPT
            formatted_prompt = prompt.format(ticket_content=ticket_content)
        elif chat_history:
            prompt = self.CONVERSATIONAL_INTAKE_PROMPT
            history_str = ""
            for turn in chat_history:
                role = "Customer" if turn.get("role") == "user" else "Agent"
                history_str += f"{role}: {turn.get('content', '')}\n"
            formatted_prompt = prompt.format(
                history_formatted=history_str.strip(),
                ticket_content=ticket_content
            )
        else:
            prompt = self.INTAKE_PROMPT
            formatted_prompt = prompt.format(ticket_content=ticket_content)
        
        try:
            response_text = await call_fast(formatted_prompt, max_tokens=256)
            
            logger.debug(f"Fast model response received")
            
            # Parse JSON
            return self._parse_json(response_text)
            
        except Exception as e:
            logger.error(f"Error calling fast LLM: {e}")
            return None

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        """
        Parse JSON from model response.
        
        Handles common issues like markdown code blocks.
        
        Args:
            text: Raw response text.
            
        Returns:
            Parsed dict or None if parsing failed.
        """
        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Also handle ```json prefix without closing
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Failed to parse: {text[:200]}")
            return None

    def _validate_complexity(self, value: Any) -> ComplexityRoute:
        """Validate and normalize complexity value."""
        if value in ("simple", "moderate", "complex"):
            return value
        return "moderate"

    def _validate_urgency(self, value: Any) -> UrgencyLevel:
        """Validate and normalize urgency value."""
        if value in ("low", "medium", "high", "critical"):
            return value
        return "medium"

    def _validate_topic(self, value: Any) -> PrimaryTopic:
        """Validate and normalize primary topic value."""
        valid_topics = ("webhook", "billing", "connect", "auth", "api", "radar", "other")
        if value in valid_topics:
            return value
        return "other"

    def _validate_error_codes(self, value: Any) -> list[str]:
        """Validate and normalize error codes list."""
        if isinstance(value, list):
            return [str(code) for code in value if code]
        return []

    def _validate_confidence(self, value: Any) -> float:
        """Validate and normalize confidence score."""
        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return 0.5

    def _apply_defaults(self, state: TicketState) -> TicketState:
        """
        Apply safe defaults when classification fails.
        
        Args:
            state: Current ticket state.
            
        Returns:
            State with default classification values.
        """
        state["complexity"] = "moderate"
        state["urgency"] = "medium"
        state["primary_topic"] = "other"
        state["error_codes"] = []
        state["search_keywords"] = []
        state["intake_confidence"] = 0.5
        state["topic_shift"] = False
        state["rewritten_query"] = state.get("ticket_content", "")
        state["agent_path"] = ["intake"]
        
        # Log error
        state["error_log"] = [
            "intake: classification failed, using defaults"
        ]
        
        return state

    def route(self, state: TicketState) -> str:
        """
        Determine routing based on intake classification.
        
        Args:
            state: Ticket state with intake classification.
            
        Returns:
            Route name: 'simple_retrieval', 'parallel_retrieval', or 'escalate'.
        """
        return self.router.route(state)


# Module-level instance
intake_agent = IntakeAgent(router=ConfidenceAdaptiveRouter())
