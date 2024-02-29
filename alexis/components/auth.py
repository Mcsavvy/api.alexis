"""Alexis dependencies."""


from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from alexis.models import User

security = HTTPBearer()


def get_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get token from header."""
    if not creds.scheme.lower() == "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
        )
    return creds.credentials


def is_authenticated(token: str = Depends(get_token)) -> "User":
    """Check if user is authenticated."""
    user = User.get_by_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    return user
