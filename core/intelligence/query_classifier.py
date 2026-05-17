"""
Query classification for adaptive retrieval weighting.

Classifies queries into 3 types:
- API: Contains Stripe API names/methods (stripe_customer_create, payment_intent)
- Problem: Contains error keywords (error, timeout, failed, webhook issue)
- Conceptual: Contains design keywords (when, why, how, best practice)
"""

import logging
import re
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification types."""
    API = "api"
    PROBLEM = "problem"
    CONCEPTUAL = "conceptual"


class ClassificationResult(NamedTuple):
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    matched_patterns: list[str]


class QueryClassifier:
    """Classifies queries for adaptive retrieval weighting."""
    
    API_PATTERNS = [
        r'\bstripe[_\.]?\w+\b',
        r'\b(?:create|list|retrieve|update|delete)_\w+\b',
        r'\b(?:customer|charge|refund|payment_intent|invoice|subscription)\b',
        r'\b(?:webhook|endpoint|event_type|api_key)\b',
    ]
    
    PROBLEM_PATTERNS = [
        r'\b(?:error|timeout|failed|failure|issue|bug|crash|broken)\b',
        r'\b(?:webhook\s+(?:not\s+)?(?:received|firing|timeout))\b',
        r'\b(?:fix|debug|troubleshoot|resolve)\b',
        r'\b(?:authentication|permission|401|403|404|500|502|503)\b',
        r'\b(?:deprecated|no\s+longer|sunset)\b',
    ]
    
    CONCEPTUAL_PATTERNS = [
        r'\b(?:when|why|how|should|can).*?\b(?:use|choose|implement|design)\b',
        r'\b(?:best\s+practice|recommended|guideline)\b',
        r'\b(?:design|architecture|pattern|workflow)\b',
        r'\b(?:difference\s+between|comparison|vs)\b',
    ]
    
    def __init__(self):
        """Initialize with precompiled regex patterns."""
        self._api = [re.compile(p, re.IGNORECASE) for p in self.API_PATTERNS]
        self._problem = [re.compile(p, re.IGNORECASE) for p in self.PROBLEM_PATTERNS]
        self._conceptual = [re.compile(p, re.IGNORECASE) for p in self.CONCEPTUAL_PATTERNS]
    
    def classify(self, query: str) -> ClassificationResult:
        """Classify query and return type with confidence."""
        q = query.lower()
        
        api_matches = sum(1 for p in self._api if p.search(q))
        problem_matches = sum(1 for p in self._problem if p.search(q))
        conceptual_matches = sum(1 for p in self._conceptual if p.search(q))
        
        if api_matches >= problem_matches and api_matches >= conceptual_matches and api_matches > 0:
            query_type = QueryType.API
            confidence = api_matches / (api_matches + max(problem_matches, conceptual_matches))
        elif problem_matches >= conceptual_matches and problem_matches > 0:
            query_type = QueryType.PROBLEM
            confidence = problem_matches / (problem_matches + max(api_matches, conceptual_matches))
        else:
            query_type = QueryType.CONCEPTUAL
            total = api_matches + problem_matches
            confidence = conceptual_matches / (conceptual_matches + total) if total > 0 else 0.33
        
        confidence = min(1.0, max(0.33, confidence))
        
        logger.info(f"[CLASSIFY] Query type={query_type.value} confidence={confidence:.2f}")
        
        return ClassificationResult(query_type, confidence, [])
    
    def get_weights(self, query_type: QueryType) -> tuple[float, float]:
        """Get adaptive RRF weights (fts_weight, vector_weight)."""
        weights = {
            QueryType.API: (0.80, 0.20),
            QueryType.PROBLEM: (0.60, 0.40),
            QueryType.CONCEPTUAL: (0.30, 0.70),
        }
        return weights.get(query_type, (0.50, 0.50))
