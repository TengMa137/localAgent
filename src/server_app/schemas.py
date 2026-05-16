from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["admin", "user"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=1024)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=1024)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
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
    title: str | None = Field(default=None, max_length=160)


class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
