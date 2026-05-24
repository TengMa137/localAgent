from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


Role = Literal["admin", "user"]
Username = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
RequiredTitle = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
OptionalTitle = Annotated[str, StringConstraints(strip_whitespace=True, max_length=160)]
MessageContent = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


class LoginRequest(BaseModel):
    username: Username
    password: str = Field(min_length=1, max_length=1024)


class RegisterRequest(BaseModel):
    username: Username
    password: str = Field(min_length=8, max_length=1024)


class CreateUserRequest(BaseModel):
    username: Username
    password: str = Field(min_length=8, max_length=1024)
    role: Role = "user"


class UpdateUserRequest(BaseModel):
    role: Role | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=1024)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=8, max_length=1024)


class CreateChatRequest(BaseModel):
    title: OptionalTitle | None = None


class RenameChatRequest(BaseModel):
    title: RequiredTitle


class SendMessageRequest(BaseModel):
    content: MessageContent
