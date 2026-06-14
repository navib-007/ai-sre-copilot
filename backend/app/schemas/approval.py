"""
schemas/approval.py — Approval Request Schemas
==============================================
CONCEPT: Pydantic Validation & Serialization
  Provides Pydantic schemas for Approval request/response payloads
  to ensure strict typing and automatic serialization of details.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator
import json


class ApprovalRequestResponse(BaseModel):
    """Schema representing an approval request returned by the API."""

    id: int
    incident_id: Optional[int] = None
    langgraph_thread_id: Optional[str] = None
    action_type: str
    action_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed dictionary containing details of the action.",
    )
    status: str
    requested_by: int
    reviewed_by: Optional[int] = None
    review_comment: Optional[str] = None
    created_at: datetime
    reviewed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

    @model_validator(mode="before")
    @classmethod
    def parse_action_details(cls, data: Any) -> Any:
        """
        Custom validator to automatically deserialize action_details_json
        from database to a dictionary if needed.
        """
        if hasattr(data, "action_details_json") and isinstance(data.action_details_json, str):
            try:
                # Set action_details attribute so model_validate maps it
                data.action_details = json.loads(data.action_details_json)
            except Exception:
                data.action_details = {}
        elif isinstance(data, dict) and "action_details_json" in data:
            try:
                data["action_details"] = json.loads(data["action_details_json"])
            except Exception:
                data["action_details"] = {}
        return data


class ApprovalDecision(BaseModel):
    """Request payload to approve or reject a pending request."""

    comment: Optional[str] = Field(
        default=None,
        description="Optional justification/comment from the reviewer.",
        max_length=500,
    )
