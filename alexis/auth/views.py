"""Auth Views."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from alexis.auth.models import User
from alexis.components import session
from alexis.dependencies import is_authenticated

router = APIRouter(prefix="/auth", tags=["auth"])


class UserDetailSchema(BaseModel):
    """User details."""

    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    picture: str | None = None


class CreateUserSchema(UserDetailSchema):
    """User schema."""

    kinde_user: str | None


class AuthTokenSchema(BaseModel):
    """Auth token schema."""

    token: str


@router.post("/authenticate")
async def authenticate(data: CreateUserSchema) -> AuthTokenSchema:
    """Authenticate."""
    if data.kinde_user:
        user = (
            session.query(User)
            .filter(User.kinde_user == data.kinde_user)
            .first()
        )
        if user:
            return AuthTokenSchema(token=user.create_token())
    try:
        user = User.create(session, **data.model_dump())
    except User.CreateError as err:
        raise HTTPException(status_code=400, detail=str(err))
    return AuthTokenSchema(token=user.create_token())  # type: ignore[union-attr]

