"""Browser binding for personal provider consent; account authority lives in knowledge."""

from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from server.auth.bearer import missing_bearer, verify_bearer_header
from server.auth.oauth_store import PENDING_TTL, token_id
from server.clients.connected_oauth import OAuthError
from server.core import request_work
from server.core.diagnostics import private_work
from server.knowledge.app import connected_accounts as accounts

router = APIRouter()


def _cookie(state: str) -> str:
    return "loom_connection_" + token_id(state)[:16]


@router.post("/v1/connect/{kind}/start")
async def start(kind: str, request: Request) -> JSONResponse:
    auth = verify_bearer_header(request.headers.get("Authorization"))
    if auth is None:
        raise missing_bearer()
    # This route never applies an on-behalf header, including from a trusted caller.
    if request.headers.get("X-Loom-On-Behalf-Of"):
        raise accounts.unavailable()
    url, state, binding = await accounts.start(auth, kind)
    redirect = urlsplit(accounts.client(kind).redirect_uri)
    response = JSONResponse({"authorization_url": url}, headers={"Cache-Control": "no-store"})
    response.set_cookie(
        _cookie(state),
        binding,
        max_age=PENDING_TTL,
        path=redirect.path,
        secure=redirect.scheme == "https",
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/v1/connect/{kind}/callback")
async def callback(kind: str, request: Request) -> JSONResponse:
    # Preserve the privacy mark through middleware and access-log completion,
    # including malformed requests whose URL still contains a code or state.
    with request_work.bind(request_work.RequestWork()):
        request_work.mark_mirrored()
        with private_work():
            try:
                return await _callback(kind, request)
            except (OAuthError, TimeoutError):
                raise accounts.unavailable() from None


async def _callback(kind: str, request: Request) -> JSONResponse:
    query = request.query_params
    if any(len(query.getlist(key)) > 1 for key in ("state", "code", "error")):
        raise accounts.unavailable()
    state = query.get("state", "")
    await accounts.callback(
        kind,
        state,
        request.cookies.get(_cookie(state), ""),
        query.get("code", ""),
        denied=bool(query.get("error")),
    )
    response = JSONResponse({"connected": True}, headers={"Cache-Control": "no-store"})
    redirect = urlsplit(accounts.client(kind).redirect_uri)
    response.delete_cookie(_cookie(state), path=redirect.path)
    return response


def attach(app: FastAPI) -> None:
    app.include_router(router)
