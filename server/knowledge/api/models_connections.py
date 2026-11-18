"""The caller's own connected accounts; credentials are never part of the wire shape."""

from typing import Literal

from pydantic import BaseModel, Field


class ConnectionListInput(BaseModel):
    org: str = Field(description="The caller's organization.")


class ConnectionEntry(BaseModel):
    kind: Literal["gdrive", "notion", "linear"]
    connected: bool
    account_email: str | None = None
    scopes: list[str] = Field(default_factory=list)
    connect_url: str


class ConnectionListOutput(BaseModel):
    connections: list[ConnectionEntry]


class ConnectionDisconnectInput(BaseModel):
    org: str = Field(description="The caller's organization.")
    kind: Literal["gdrive", "notion", "linear"] = Field(
        description="The connected provider to disconnect."
    )


class ConnectionDisconnectOutput(BaseModel):
    disconnected: bool
