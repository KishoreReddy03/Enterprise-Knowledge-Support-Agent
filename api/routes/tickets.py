"""
Ticket processing endpoints.

Handles incoming support tickets and returns AI-generated responses
with confidence scores and rep guidance.
"""

import logging
import time
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Request
from langfuse import observe
from pydantic import BaseModel, Field

from config import settings
from core.agents.orchestrator import process_ticket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tickets", tags=["tickets"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TicketProcessRequest(BaseModel):
    """Request body for ticket processing."""
    
    ticket_content: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The support ticket content to process",
    )
    customer_id: str | None = Field(
        default=None,
        description="Optional customer identifier",
    )
    customer_tier: Literal["free", "standard", "enterprise"] = Field(
        default="standard",
        description="Customer tier for prioritization",
    )


class SourceCited(BaseModel):
    """A source cited in the response."""
    
    title: str
    url: str
    relevance_score: float


class TicketProcessResponse(BaseModel):
    """Response from ticket processing."""
    
    ticket_id: str = Field(description="Unique ticket identifier")
    draft_reply: str = Field(description="AI-generated reply text")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the response (0-1)",
    )
    rep_guidance: str = Field(description="Guidance for the support rep")
    sources_cited: list[SourceCited] = Field(
        default_factory=list,
        description="Sources used in generating the response",
    )
    escalation_needed: bool = Field(
        default=False,
        description="Whether human escalation is recommended",
    )
    escalation_brief: str | None = Field(
        default=None,
        description="Brief for escalation if needed",
    )
    processing_time_ms: int = Field(
        description="Total processing time in milliseconds",
    )
    agent_path: list[str] = Field(
        default_factory=list,
        description="Sequence of agents that processed this ticket",
    )


class TicketErrorResponse(BaseModel):
    """Error response for ticket processing."""
    
    error: str = Field(description="Error message")
    code: str = Field(description="Error code")
    ticket_id: str | None = Field(
        default=None,
        description="Ticket ID if assigned before error",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/process",
    response_model=TicketProcessResponse,
    responses={
        400: {"model": TicketErrorResponse},
        500: {"model": TicketErrorResponse},
    },
    summary="Process a support ticket",
    description="Submit a support ticket for AI-powered response generation",
)
@observe(name="api_process_ticket")
async def process_ticket_endpoint(
    request: Request,
    body: TicketProcessRequest,
) -> TicketProcessResponse:
    """
    Process a support ticket through the AI pipeline.
    
    Returns a draft reply with confidence score and rep guidance.
    """
    ticket_id = str(uuid4())
    start_time = time.time()
    
    # Get request ID from middleware
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.info(
        f"Processing ticket {ticket_id} | "
        f"request_id={request_id} | "
        f"tier={body.customer_tier}"
    )
    
    try:
        # Update Langfuse context
        # langfuse_context.update_current_trace(
        #     name="ticket_processing",
        #     user_id=body.customer_id or "anonymous",
        #     metadata={
        #         "ticket_id": ticket_id,
        #         "customer_tier": body.customer_tier,
        #         "request_id": request_id,
        #     },
        # )
        
        # Process through orchestrator
        result = await process_ticket(
            ticket_content=body.ticket_content,
            customer_id=body.customer_id or "unknown",
            customer_tier=body.customer_tier,
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Extract response data
        final_response = result.get("final_response", {})
        
        # Format sources
        sources_cited = []
        raw_sources = final_response.get("sources", [])
        for src in raw_sources:
            sources_cited.append(SourceCited(
                title=src.get("title", "Unknown"),
                url=src.get("url", ""),
                relevance_score=src.get("relevance_score", 0.0),
            ))
        
        logger.info(
            f"Ticket {ticket_id} processed | "
            f"time_ms={processing_time_ms} | "
            f"confidence={result.get('confidence_score', 0.0):.2f} | "
            f"escalated={result.get('escalated', False)}"
        )
        
        return TicketProcessResponse(
            ticket_id=ticket_id,
            draft_reply=final_response.get("reply_text", ""),
            confidence_score=result.get("confidence_score", 0.0),
            rep_guidance=final_response.get("rep_guidance", "Review before sending."),
            sources_cited=sources_cited,
            escalation_needed=result.get("escalated", False),
            escalation_brief=result.get("escalation_brief"),
            processing_time_ms=processing_time_ms,
            agent_path=result.get("agent_path", []),
        )
        
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Error processing ticket {ticket_id}: {e}",
            exc_info=True,
        )
        
        # Return error response
        raise
