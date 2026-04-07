"""
Feedback capture endpoints.

Captures rep edits and ratings to enable system learning.
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel, Field, field_validator

from core.intelligence.feedback_loop import FeedbackProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    """Request body for submitting feedback."""
    
    response_id: str = Field(
        ...,
        description="ID of the AI response being reviewed",
    )
    original_reply: str = Field(
        ...,
        description="The original AI-generated reply",
    )
    edited_reply: str = Field(
        ...,
        description="The rep's edited version (or same as original if unchanged)",
    )
    was_sent: bool = Field(
        ...,
        description="Whether the reply was sent to the customer",
    )
    rep_rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Optional rep rating 1-5",
    )
    
    @field_validator("edited_reply")
    @classmethod
    def validate_edited_reply(cls, v: str) -> str:
        """Ensure edited reply is not empty."""
        if not v.strip():
            raise ValueError("edited_reply cannot be empty")
        return v


class FeedbackResponse(BaseModel):
    """Response from feedback submission."""
    
    feedback_id: str = Field(description="Unique feedback record ID")
    edit_type: str = Field(
        description="Type of edit detected (none, factual_correction, tone_adjustment, etc.)"
    )


class FeedbackErrorResponse(BaseModel):
    """Error response for feedback."""
    
    error: str = Field(description="Error message")
    code: str = Field(description="Error code")
    response_id: str | None = Field(
        default=None,
        description="Response ID from request if available",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "",
    response_model=FeedbackResponse,
    responses={
        400: {"model": FeedbackErrorResponse},
        500: {"model": FeedbackErrorResponse},
    },
    summary="Submit feedback on an AI response",
    description="Capture rep edits and ratings to improve the system",
)
@observe(name="api_submit_feedback")
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
) -> FeedbackResponse:
    """
    Submit feedback on an AI-generated response.
    
    Captures edits for learning and flags problematic sources.
    """
    feedback_id = str(uuid4())
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.info(
        f"Receiving feedback for response {body.response_id} | "
        f"request_id={request_id} | "
        f"was_sent={body.was_sent}"
    )
    
    try:
        # Update Langfuse context
        langfuse_context.update_current_trace(
            name="feedback_submission",
            metadata={
                "feedback_id": feedback_id,
                "response_id": body.response_id,
                "was_sent": body.was_sent,
                "has_rating": body.rep_rating is not None,
                "request_id": request_id,
            },
        )
        
        # Process feedback
        processor = FeedbackProcessor()
        record = await processor.capture_edit(
            response_id=body.response_id,
            original_reply=body.original_reply,
            edited_reply=body.edited_reply,
            was_sent=body.was_sent,
            rep_rating=body.rep_rating,
        )
        
        logger.info(
            f"Feedback {feedback_id} captured | "
            f"edit_type={record.edit_type} | "
            f"response_id={body.response_id}"
        )
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            edit_type=record.edit_type,
        )
        
    except Exception as e:
        logger.error(
            f"Error processing feedback for {body.response_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "code": "FEEDBACK_PROCESSING_ERROR",
                "response_id": body.response_id,
            },
        )


@router.get(
    "/insights",
    summary="Get improvement insights",
    description="Get weekly summary of system performance based on rep feedback",
)
@observe(name="api_get_insights")
async def get_insights(request: Request) -> dict:
    """
    Get improvement insights from feedback data.
    
    Returns summary of common edits and areas for improvement.
    """
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.info(f"Fetching insights | request_id={request_id}")
    
    try:
        processor = FeedbackProcessor()
        insights = await processor.get_improvement_insights()
        
        return insights
        
    except Exception as e:
        logger.error(f"Error fetching insights: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "code": "INSIGHTS_FETCH_ERROR",
            },
        )
