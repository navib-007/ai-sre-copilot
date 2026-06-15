"""
schemas/auth.py — Authentication Schemas
========================================
CONCEPT: Request/Response Validation for Auth Flows
"""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from app.db.models import UserRole


class UserRegister(BaseModel):
    """Schema for registering a new user."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., max_length=255, description="Unique email address")
    password: str = Field(..., min_length=6, description="Password (min 6 characters)")
    role: UserRole = Field(default=UserRole.ENGINEER, description="Assigned role: admin, engineer, viewer")


class UserLogin(BaseModel):
    """Schema for user login credentials."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Schema for successful authentication response containing JWT tokens."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")


class TokenRefresh(BaseModel):
    """Schema for requesting a new access token via refresh token."""
    refresh_token: str = Field(..., description="Valid JWT refresh token")


class UserResponse(BaseModel):
    """Schema for returning user information safely (hiding passwords)."""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
