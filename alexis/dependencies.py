"""Alexis dependencies."""


from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from alexis.auth.models import User
from alexis.components import db

security = HTTPBearer()


async def get_session():
    """SQLAlchemy session dependency."""
    with db.session() as session:
        yield session


def get_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get token from header."""
    if not creds.scheme.lower() == "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
        )
    return creds.credentials


def is_authenticated(
    session: Session = Depends(get_session), token: str = Depends(get_token)
) -> "User":
    """Check if user is authenticated."""
    user = User.get_by_token(session, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    return user


session_dependency = Depends(get_session)
