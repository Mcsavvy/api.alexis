"""Alexis dependencies."""


from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sentry_sdk import set_user

from alexis.models import User

security = HTTPBearer()


async def get_token(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """Get token from header."""
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    if not creds.scheme.lower() == "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
        )
    return creds.credentials


async def is_authenticated(token: str = Depends(get_token)) -> "User":
    """Check if user is authenticated."""
    user = User.get_by_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    set_user({"id": user.uid, "username": user.name, "email": user.email})
    return user
