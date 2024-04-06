"""Auth Views."""


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from alexis.components.auth import is_authenticated
from alexis.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class UserDetailSchema(BaseModel):
    """User details."""

    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    picture: str | None = None

    class Config:
        """Pydantic config."""

        from_attributes = True


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
        user = User.objects.filter(kinde_user=data.kinde_user).one()
        if user:
            return AuthTokenSchema(token=user.create_token())
    try:
        user = User.create(**data.model_dump())
    except User.CreateError as err:
        raise HTTPException(status_code=400, detail=str(err))
    return AuthTokenSchema(token=user.create_token())  # type: ignore[union-attr]


@router.get("/me", response_model=UserDetailSchema)
async def get_me(user: User = Depends(is_authenticated)) -> UserDetailSchema:
    """Get me."""
    return user
