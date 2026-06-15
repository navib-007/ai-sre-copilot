"""
auth/rbac.py — Role-Based Access Control Dependencies
=====================================================
CONCEPT: Request Authorization & Role Enforcement
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Sequence

from app.auth.jwt_handler import decode_token
from app.db.database import get_db
from app.db.models import User, UserRole

# Define the token URL which Swagger UI will use to login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency to retrieve the currently logged-in user from the JWT token.
    Raises HTTP 401 if authorization fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    payload = decode_token(token)
    if not payload:
        raise credentials_exception
        
    username: str | None = payload.get("sub")
    token_type: str | None = payload.get("type")
    
    if not username or token_type != "access":
        raise credentials_exception
        
    # Query database for user
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    if not user:
        raise credentials_exception
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
        
    return user


class RoleChecker:
    """
    FastAPI dependency factory to enforce required roles on endpoints.
    """
    def __init__(self, allowed_roles: Sequence[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted for this user role"
            )
        return user


# Convenient Role Checker Dependency instances
require_admin = RoleChecker(["admin"])
require_engineer = RoleChecker(["admin", "engineer"])
require_viewer = RoleChecker(["admin", "engineer", "viewer"])
