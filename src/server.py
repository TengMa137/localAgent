from __future__ import annotations

import asyncio
import mimetypes
import secrets
import shutil
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic_ai.messages import ModelRequest as ModelRequest, ModelResponse as ModelResponse, TextPart as TextPart

from run_agents import ChatSession, run_turn
from server_app.chat_store import (
    active_branch,
    agent_lock_key,
    decorate_branch_variants,
    insert_assistant_placeholder,
    insert_user_message,
    load_branch_for_user,
    mark_stale_running_messages,
    messages_from_json,
    messages_to_json,
    prompt_with_session_context,
    rows_to_model_history,
    session_upload_context,
    visible_message_rows,
)
from server_app.schemas import (
    ChangePasswordRequest,
    CreateChatRequest,
    CreateUserRequest,
    LoginRequest,
    RegisterRequest,
    RenameChatRequest,
    SendMessageRequest,
    UpdateUserRequest,
)
from server_app.serializers import (
    compact_trace_events,
    compact_turn_logs,
    message_metadata,
    public_chat_file,
    public_chat_session,
    public_message,
    public_user,
)
from server_app.utils import (
    clean_filename,
    is_text_preview,
    json_dumps,
    row_has,
    slugish,
    sqlite_lastrowid,
    utc_now,
)

_json_dumps = json_dumps
_utc_now = utc_now


class Settings(BaseSettings):
    state_dir: Path = Path("./localagent_state")
    db_path: Path | None = None
    web_dir: Path = Path(__file__).resolve().parent.parent / "web"
    docs_dir: Path = Path("./user_docs")
    max_upload_bytes: int = 25 * 1024 * 1024
    session_cookie: str = "localagent_session"
    cookie_secure: bool = False
    session_ttl_seconds: int = 7 * 24 * 3600
    admin_username: str = ""
    admin_password: str = ""

    model_config = SettingsConfigDict(
        env_prefix="LOCALAGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
STATE_DIR = settings.state_dir
DB_PATH = settings.db_path or STATE_DIR / "localagent.sqlite3"
WEB_DIR = settings.web_dir
DOCS_DIR = settings.docs_dir
WEB_UPLOAD_ROOT = DOCS_DIR / "web_uploads"
SESSION_COOKIE = settings.session_cookie
CSRF_COOKIE = "localagent_csrf"
COOKIE_SECURE = settings.cookie_secure
SESSION_TTL_SECONDS = settings.session_ttl_seconds
LOGIN_WINDOW_SECONDS = 300
LOGIN_MAX_ATTEMPTS = 8
TEXT_PREVIEW_LIMIT = 1_000_000

ph = PasswordHasher()
_login_attempts: dict[str, list[float]] = {}
_agent_locks: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="Local Agent", version="0.1.0", lifespan=lifespan)


def _upload_virtual_path(user_id: int, session_id: str, stored_name: str) -> str:
    return f"/docs/web_uploads/{user_id}/{session_id}/{stored_name}"


def _upload_host_path(user_id: int, session_id: str, stored_name: str) -> Path:
    return WEB_UPLOAD_ROOT / str(user_id) / session_id / stored_name


def _client_key(request: Request, username: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{username.strip().lower()}"


def _check_login_rate(request: Request, username: str) -> None:
    key = _client_key(request, username)
    now = time.time()
    window = [ts for ts in _login_attempts.get(key, []) if now - ts < LOGIN_WINDOW_SECONDS]
    if len(window) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many login attempts.")
    _login_attempts[key] = window


def _record_login_failure(request: Request, username: str) -> None:
    key = _client_key(request, username)
    _login_attempts.setdefault(key, []).append(time.time())


def _clear_login_failures(request: Request, username: str) -> None:
    _login_attempts.pop(_client_key(request, username), None)


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS web_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                csrf_token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                model_messages_json TEXT NOT NULL DEFAULT '[]',
                active_branch_id TEXT NOT NULL DEFAULT 'main',
                archived_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_branches (
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                id TEXT NOT NULL,
                parent_id TEXT,
                fork_parent_message_id INTEGER REFERENCES chat_messages(id) ON DELETE SET NULL,
                variant_number INTEGER NOT NULL DEFAULT 1,
                model_messages_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, id)
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                fork_parent_id INTEGER REFERENCES chat_messages(id) ON DELETE SET NULL,
                variant_number INTEGER NOT NULL DEFAULT 1,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                virtual_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                content_type TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            """
        )
        session_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()}
        if "active_branch_id" not in session_columns:
            conn.execute("ALTER TABLE chat_sessions ADD COLUMN active_branch_id TEXT NOT NULL DEFAULT 'main'")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}
        if "metadata_json" not in columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
        if "branch_id" not in columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN branch_id TEXT NOT NULL DEFAULT 'main'")
        if "fork_parent_id" not in columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN fork_parent_id INTEGER REFERENCES chat_messages(id) ON DELETE SET NULL")
        if "variant_number" not in columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN variant_number INTEGER NOT NULL DEFAULT 1")
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_branches (
                session_id, user_id, id, parent_id, fork_parent_message_id,
                variant_number, model_messages_json, created_at, updated_at
            )
            SELECT id, user_id, 'main', NULL, NULL, 1, model_messages_json, created_at, updated_at
            FROM chat_sessions
            """
        )
        count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        if count == 0:
            username = settings.admin_username.strip()
            password = settings.admin_password
            if username and password:
                conn.execute(
                    """
                    INSERT INTO users (username, password_hash, role, is_active, created_at)
                    VALUES (?, ?, 'admin', 1, ?)
                    """,
                    (username, ph.hash(password), utc_now()),
                )


def _audit(conn: sqlite3.Connection, user_id: int | None, event_type: str, details: dict[str, Any] | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_events (user_id, event_type, details_json, created_at) VALUES (?, ?, ?, ?)",
        (user_id, event_type, json_dumps(details or {}), utc_now()),
    )


def _user_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"])


def _create_web_session(conn: sqlite3.Connection, *, user_id: int) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    conn.execute(
        """
        INSERT INTO web_sessions (token, user_id, csrf_token, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (token, user_id, csrf_token, utc_now(), int(time.time()) + SESSION_TTL_SECONDS),
    )
    return token, csrf_token


def _set_session_cookies(response: Response, token: str, csrf_token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        max_age=SESSION_TTL_SECONDS,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _clear_session_cookies(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


def _revoke_user_sessions(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("DELETE FROM web_sessions WHERE user_id = ?", (user_id,))


def _get_current_user(request: Request) -> sqlite3.Row:
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
    now = int(time.time())
    with db() as conn:
        row = conn.execute(
            """
            SELECT u.*, s.csrf_token, s.expires_at
            FROM web_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if not row or row["expires_at"] < now or not row["is_active"]:
            if row:
                conn.execute("DELETE FROM web_sessions WHERE token = ?", (token,))
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
        return row


def current_user(request: Request) -> sqlite3.Row:
    user = _get_current_user(request)
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        expected = user["csrf_token"]
        supplied = request.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(str(expected), supplied):
            raise HTTPException(status_code=403, detail="Invalid CSRF token.")
    return user


def require_admin(user: sqlite3.Row = Depends(current_user)) -> sqlite3.Row:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


def _load_chat_for_user(conn: sqlite3.Connection, session_id: str, user_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM chat_sessions WHERE id = ? AND user_id = ? AND archived_at IS NULL",
        (session_id, user_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return row


def _chat_session_for_agent(
    row: sqlite3.Row,
    user_id: int,
    *,
    model_messages_json: str | None = None,
    branch_id: str = "main",
) -> ChatSession:
    history_dir = STATE_DIR / "history" / str(user_id)
    report_dir = STATE_DIR / "reports" / str(user_id) / row["id"] / branch_id
    return ChatSession(
        message_history=messages_from_json(model_messages_json if model_messages_json is not None else row["model_messages_json"]),
        session_title=row["title"],
        history_path=history_dir / f"{row['id']}-{branch_id}.json",
        report_dir=report_dir,
    )


def _load_chat_file_for_user(conn: sqlite3.Connection, file_id: int, session_id: str, user_id: int) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT *
        FROM chat_files
        WHERE id = ? AND session_id = ? AND user_id = ?
        """,
        (file_id, session_id, user_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="File not found.")
    return row


def _list_session_reports(session: sqlite3.Row, user_id: int) -> list[dict[str, Any]]:
    branch_id = session["active_branch_id"] if row_has(session, "active_branch_id") else "main"
    report_dir = _chat_session_for_agent(session, user_id, branch_id=branch_id).report_dir
    if report_dir is None or not report_dir.exists():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*-report.md")):
        stat = path.stat()
        reports.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return reports


def _session_report_path(session: sqlite3.Row, user_id: int, report_name: str) -> Path:
    name = Path(report_name).name
    if name != report_name or not name.endswith("-report.md"):
        raise HTTPException(status_code=404, detail="Report not found.")
    branch_id = session["active_branch_id"] if row_has(session, "active_branch_id") else "main"
    report_dir = _chat_session_for_agent(session, user_id, branch_id=branch_id).report_dir
    if report_dir is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    path = (report_dir / name).resolve()
    root = report_dir.resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise HTTPException(status_code=404, detail="Report not found.")
    return path


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register")
def register(body: RegisterRequest, response: Response) -> dict[str, Any]:
    with db() as conn:
        if _user_count(conn) != 0:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO users (username, password_hash, role, is_active, created_at)
                    VALUES (?, ?, 'user', 1, ?)
                    """,
                    (body.username.strip(), ph.hash(body.password), utc_now()),
                )
            except sqlite3.IntegrityError:
                raise HTTPException(status_code=409, detail="Username already exists.")
        else:
            raise HTTPException(
                status_code=409,
                detail="Admin account is not initialized. Set LOCALAGENT_ADMIN_USERNAME and LOCALAGENT_ADMIN_PASSWORD on the backend.",
            )
        user = conn.execute("SELECT * FROM users WHERE id = ?", (sqlite_lastrowid(cur),)).fetchone()
        token, csrf_token = _create_web_session(conn, user_id=user["id"])
        _audit(conn, user["id"], "register_user", {})
        _set_session_cookies(response, token, csrf_token)
        return {"user": public_user(user)}


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    _check_login_rate(request, body.username)
    with db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (body.username.strip(),),
        ).fetchone()
        if not user or not user["is_active"]:
            _record_login_failure(request, body.username)
            _audit(conn, None, "login_failed", {"username": body.username.strip(), "reason": "unknown_user"})
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        try:
            ph.verify(user["password_hash"], body.password)
        except VerifyMismatchError:
            _record_login_failure(request, body.username)
            _audit(conn, user["id"], "login_failed", {"username": user["username"], "reason": "bad_password"})
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        if ph.check_needs_rehash(user["password_hash"]):
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (ph.hash(body.password), user["id"]))
        token, csrf_token = _create_web_session(conn, user_id=user["id"])
        _audit(conn, user["id"], "login_success", {})
        _clear_login_failures(request, body.username)
        _set_session_cookies(response, token, csrf_token)
        return {"user": public_user(user)}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user: sqlite3.Row = Depends(current_user)) -> dict[str, bool]:
    token = request.cookies.get(SESSION_COOKIE, "")
    with db() as conn:
        if token:
            conn.execute("DELETE FROM web_sessions WHERE token = ?", (token,))
        _audit(conn, user["id"], "logout", {})
    _clear_session_cookies(response)
    return {"ok": True}


@app.get("/api/me")
def me(user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    return {"user": public_user(user)}


@app.patch("/api/me/password")
def change_my_password(
    body: ChangePasswordRequest,
    response: Response,
    user: sqlite3.Row = Depends(current_user),
) -> dict[str, bool]:
    try:
        ph.verify(user["password_hash"], body.current_password)
    except VerifyMismatchError:
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    with db() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (ph.hash(body.new_password), user["id"]))
        _revoke_user_sessions(conn, user["id"])
        _audit(conn, user["id"], "change_own_password", {})
    _clear_session_cookies(response)
    return {"ok": True}


@app.get("/api/admin/users")
def list_users(user: sqlite3.Row = Depends(require_admin)) -> dict[str, Any]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        return {"users": [public_user(row) for row in rows]}


@app.post("/api/admin/users")
def create_user(body: CreateUserRequest, admin: sqlite3.Row = Depends(require_admin)) -> dict[str, Any]:
    with db() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO users (username, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, 1, ?)
                """,
                (body.username.strip(), ph.hash(body.password), body.role, utc_now()),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Username already exists.")
        row = conn.execute("SELECT * FROM users WHERE id = ?", (sqlite_lastrowid(cur),)).fetchone()
        _audit(conn, admin["id"], "admin_create_user", {"target_user_id": row["id"], "role": row["role"]})
        return {"user": public_user(row)}


@app.patch("/api/admin/users/{user_id}")
def update_user(
    user_id: int,
    body: UpdateUserRequest,
    response: Response,
    admin: sqlite3.Row = Depends(require_admin),
) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found.")
        if body.role is not None:
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (body.role, user_id))
        if body.is_active is not None:
            conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if body.is_active else 0, user_id))
            if not body.is_active:
                _revoke_user_sessions(conn, user_id)
        if body.password is not None:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (ph.hash(body.password), user_id))
            _revoke_user_sessions(conn, user_id)
        _audit(conn, admin["id"], "admin_update_user", {"target_user_id": user_id})
        updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if user_id == admin["id"] and (body.password is not None or body.is_active is False):
            _clear_session_cookies(response)
        return {"user": public_user(updated)}


@app.get("/api/admin/chat/sessions")
def list_all_chat_sessions(admin: sqlite3.Row = Depends(require_admin)) -> dict[str, Any]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.user_id, u.username, s.title, s.active_branch_id, s.created_at, s.updated_at
            FROM chat_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.archived_at IS NULL
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
        _audit(conn, admin["id"], "admin_list_chat_sessions", {})
        return {"sessions": [public_chat_session(row) for row in rows]}


@app.get("/api/admin/chat/sessions/{session_id}")
def get_any_chat_session(session_id: str, admin: sqlite3.Row = Depends(require_admin)) -> dict[str, Any]:
    with db() as conn:
        session = conn.execute(
            """
            SELECT s.id, s.user_id, u.username, s.title, s.active_branch_id, s.created_at, s.updated_at
            FROM chat_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.archived_at IS NULL
            """,
            (session_id,),
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        messages = conn.execute(
            "SELECT id, role, content, metadata_json, created_at, branch_id, fork_parent_id, variant_number FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        _audit(conn, admin["id"], "admin_view_chat_session", {"session_id": session_id, "user_id": session["user_id"]})
        return {
            "session": public_chat_session(session),
            "messages": [public_message(row) for row in messages],
        }


@app.get("/api/chat/sessions")
def list_chat_sessions(user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, title, active_branch_id, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ? AND archived_at IS NULL
            ORDER BY updated_at DESC
            """,
            (user["id"],),
        ).fetchall()
        return {"sessions": [public_chat_session(row) for row in rows]}


@app.post("/api/chat/sessions")
def create_chat_session(body: CreateChatRequest, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    session_id = secrets.token_urlsafe(16)
    title = (body.title or "New chat").strip() or "New chat"
    now = utc_now()
    with db() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (id, user_id, title, model_messages_json, created_at, updated_at)
            VALUES (?, ?, ?, '[]', ?, ?)
            """,
            (session_id, user["id"], title, now, now),
        )
        conn.execute(
            """
            INSERT INTO chat_branches (
                session_id, user_id, id, parent_id, fork_parent_message_id,
                variant_number, model_messages_json, created_at, updated_at
            )
            VALUES (?, ?, 'main', NULL, NULL, 1, '[]', ?, ?)
            """,
            (session_id, user["id"], now, now),
        )
        _audit(conn, user["id"], "chat_create", {"session_id": session_id})
    return {"session": {"id": session_id, "title": title, "active_branch_id": "main", "created_at": now, "updated_at": now}, "messages": []}


@app.get("/api/chat/sessions/{session_id}")
def get_chat_session(session_id: str, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    with db() as conn:
        session = _load_chat_for_user(conn, session_id, user["id"])
        mark_stale_running_messages(conn, session_id, user["id"], _agent_locks)
        branch = active_branch(conn, session, user["id"])
        message_rows = visible_message_rows(conn, session_id, user["id"], branch["id"])
        messages = [public_message(row) for row in message_rows]
        return {
            "session": {
                "id": session["id"],
                "title": session["title"],
                "active_branch_id": branch["id"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
            },
            "messages": decorate_branch_variants(
                conn,
                messages,
                session_id=session_id,
                user_id=user["id"],
                active_branch_id=branch["id"],
            ),
        }


@app.get("/api/chat/sessions/{session_id}/files")
def list_chat_files(session_id: str, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    with db() as conn:
        session = _load_chat_for_user(conn, session_id, user["id"])
        uploads = conn.execute(
            """
            SELECT *
            FROM chat_files
            WHERE session_id = ? AND user_id = ?
            ORDER BY id DESC
            """,
            (session_id, user["id"]),
        ).fetchall()
        return {
            "uploads": [public_chat_file(row) for row in uploads],
            "reports": _list_session_reports(session, user["id"]),
        }


@app.post("/api/chat/sessions/{session_id}/files")
async def upload_chat_file(
    session_id: str,
    file: UploadFile = File(...),  # noqa: B008
    user: sqlite3.Row = Depends(current_user),
) -> dict[str, Any]:
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Upload is too large.")

    filename = clean_filename(file.filename)
    stored_name = f"{int(time.time())}-{secrets.token_hex(4)}-{filename}"
    host_path = _upload_host_path(user["id"], session_id, stored_name)
    virtual_path = _upload_virtual_path(user["id"], session_id, stored_name)
    now = utc_now()

    with db() as conn:
        _load_chat_for_user(conn, session_id, user["id"])
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_bytes(content)
        cur = conn.execute(
            """
            INSERT INTO chat_files (
                session_id, user_id, filename, stored_name, virtual_path,
                size_bytes, content_type, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user["id"],
                filename,
                stored_name,
                virtual_path,
                len(content),
                file.content_type,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM chat_files WHERE id = ?", (sqlite_lastrowid(cur),)).fetchone()
        _audit(conn, user["id"], "chat_file_upload", {"session_id": session_id, "file_id": row["id"]})
        return {"file": public_chat_file(row)}


@app.get("/api/chat/sessions/{session_id}/files/{file_id}/content")
def get_chat_file_content(session_id: str, file_id: int, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    with db() as conn:
        _load_chat_for_user(conn, session_id, user["id"])
        row = _load_chat_file_for_user(conn, file_id, session_id, user["id"])

    path = _upload_host_path(user["id"], session_id, row["stored_name"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    size = path.stat().st_size
    if size <= TEXT_PREVIEW_LIMIT:
        sample = path.read_bytes()
    else:
        with path.open("rb") as fh:
            sample = fh.read(4096)
    content_type = row["content_type"] or mimetypes.guess_type(row["filename"])[0]
    is_text = size <= TEXT_PREVIEW_LIMIT and is_text_preview(row["filename"], content_type, sample)
    content = path.read_text(encoding="utf-8", errors="replace") if is_text else ""
    return {
        "file": public_chat_file(row),
        "is_text": is_text,
        "content": content,
        "raw_url": f"/api/chat/sessions/{session_id}/files/{file_id}/raw",
    }


@app.get("/api/chat/sessions/{session_id}/files/{file_id}/raw")
def get_chat_file_raw(session_id: str, file_id: int, user: sqlite3.Row = Depends(current_user)) -> FileResponse:
    with db() as conn:
        _load_chat_for_user(conn, session_id, user["id"])
        row = _load_chat_file_for_user(conn, file_id, session_id, user["id"])
    path = _upload_host_path(user["id"], session_id, row["stored_name"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = row["content_type"] or mimetypes.guess_type(row["filename"])[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=row["filename"])


@app.get("/api/chat/sessions/{session_id}/reports/{report_name}")
def get_chat_report(session_id: str, report_name: str, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    with db() as conn:
        session = _load_chat_for_user(conn, session_id, user["id"])
    path = _session_report_path(session, user["id"], report_name)
    return {
        "report": {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        },
        "content": path.read_text(encoding="utf-8", errors="replace"),
    }


@app.patch("/api/chat/sessions/{session_id}")
def rename_chat_session(session_id: str, body: RenameChatRequest, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    with db() as conn:
        _load_chat_for_user(conn, session_id, user["id"])
        now = utc_now()
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (body.title.strip(), now, session_id, user["id"]),
        )
        _audit(conn, user["id"], "chat_rename", {"session_id": session_id})
    return {"ok": True}


@app.delete("/api/chat/sessions/{session_id}")
def delete_chat_session(session_id: str, user: sqlite3.Row = Depends(current_user)) -> dict[str, bool]:
    with db() as conn:
        _load_chat_for_user(conn, session_id, user["id"])
        upload_rows = conn.execute(
            "SELECT stored_name FROM chat_files WHERE session_id = ? AND user_id = ?",
            (session_id, user["id"]),
        ).fetchall()
        report_dir = STATE_DIR / "reports" / str(user["id"]) / session_id
        conn.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user["id"]))
        _audit(conn, user["id"], "chat_delete", {"session_id": session_id})

    upload_dir = WEB_UPLOAD_ROOT / str(user["id"]) / session_id
    if upload_rows and upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    if report_dir is not None and report_dir.exists():
        shutil.rmtree(report_dir, ignore_errors=True)
    return {"ok": True}


@app.post("/api/chat/sessions/{session_id}/branches/{branch_id}/activate")
def activate_chat_branch(session_id: str, branch_id: str, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    with db() as conn:
        session = _load_chat_for_user(conn, session_id, user["id"])
        branch = load_branch_for_user(conn, session_id, branch_id, user["id"])
        mark_stale_running_messages(conn, session_id, user["id"], _agent_locks)
        now = utc_now()
        conn.execute(
            """
            UPDATE chat_sessions
            SET active_branch_id = ?, model_messages_json = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (branch["id"], branch["model_messages_json"], now, session_id, user["id"]),
        )
        _audit(conn, user["id"], "chat_branch_activate", {"session_id": session_id, "branch_id": branch["id"]})
        message_rows = visible_message_rows(conn, session_id, user["id"], branch["id"])
        messages = [public_message(row) for row in message_rows]
        return {
            "session": {
                "id": session["id"],
                "title": session["title"],
                "active_branch_id": branch["id"],
                "created_at": session["created_at"],
                "updated_at": now,
            },
            "messages": decorate_branch_variants(
                conn,
                messages,
                session_id=session_id,
                user_id=user["id"],
                active_branch_id=branch["id"],
            ),
        }


@app.post("/api/chat/sessions/{session_id}/messages/{message_id}/fork")
async def fork_chat_from_message(
    session_id: str,
    message_id: int,
    body: SendMessageRequest,
    user: sqlite3.Row = Depends(current_user),
) -> dict[str, Any]:
    with db() as conn:
        _load_chat_for_user(conn, session_id, user["id"])
        source_message = conn.execute(
            """
            SELECT id, role, content, metadata_json, created_at, branch_id, fork_parent_id, variant_number
            FROM chat_messages
            WHERE id = ? AND session_id = ? AND user_id = ? AND role = 'user'
            """,
            (message_id, session_id, user["id"]),
        ).fetchone()
        if not source_message:
            raise HTTPException(status_code=404, detail="User message not found.")

        root_message_id = int(source_message["fork_parent_id"] or source_message["id"])
        next_variant = int(
            conn.execute(
                """
                SELECT COALESCE(MAX(variant_number), 1) + 1 AS next_variant
                FROM chat_messages
                WHERE session_id = ?
                  AND user_id = ?
                  AND role = 'user'
                  AND (id = ? OR fork_parent_id = ?)
                """,
                (session_id, user["id"], root_message_id, root_message_id),
            ).fetchone()["next_variant"]
        )
        new_branch_id = secrets.token_urlsafe(10)
        now = utc_now()
        prefix_rows = visible_message_rows(
            conn,
            session_id,
            user["id"],
            source_message["branch_id"],
            before_message_id=source_message["id"],
        )
        prefix_history = messages_to_json(rows_to_model_history(prefix_rows))
        conn.execute(
            """
            INSERT INTO chat_branches (
                session_id, user_id, id, parent_id, fork_parent_message_id,
                variant_number, model_messages_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user["id"],
                new_branch_id,
                source_message["branch_id"],
                source_message["id"],
                next_variant,
                prefix_history,
                now,
                now,
            ),
        )
        cur = conn.execute(
            """
            INSERT INTO chat_messages (
                session_id, user_id, branch_id, fork_parent_id, variant_number,
                role, content, created_at
            )
            VALUES (?, ?, ?, ?, ?, 'user', ?, ?)
            """,
            (session_id, user["id"], new_branch_id, root_message_id, next_variant, body.content, now),
        )
        edited_message_id = sqlite_lastrowid(cur)
        conn.execute(
            """
            UPDATE chat_sessions
            SET active_branch_id = ?, model_messages_json = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (new_branch_id, prefix_history, now, session_id, user["id"]),
        )
        _audit(
            conn,
            user["id"],
            "chat_branch_fork",
            {
                "session_id": session_id,
                "source_message_id": source_message["id"],
                "branch_id": new_branch_id,
                "variant_number": next_variant,
            },
        )

    return await _execute_chat_message(
        session_id,
        body.content,
        user,
        branch_id=new_branch_id,
        existing_user_message_id=edited_message_id,
    )


async def _execute_chat_message(
    session_id: str,
    content: str,
    user: sqlite3.Row,
    *,
    trace_sink: Any = None,
    branch_id: str | None = None,
    existing_user_message_id: int | None = None,
) -> dict[str, Any]:
    lock_key = agent_lock_key(user["id"], session_id)
    lock = _agent_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        with db() as conn:
            session = _load_chat_for_user(conn, session_id, user["id"])
            branch = (
                load_branch_for_user(conn, session_id, branch_id, user["id"])
                if branch_id
                else active_branch(conn, session, user["id"])
            )
            now = utc_now()
            if existing_user_message_id is None:
                user_message_id = insert_user_message(
                    conn,
                    session_id=session_id,
                    user_id=user["id"],
                    branch_id=branch["id"],
                    content=content,
                    created_at=now,
                )
            else:
                user_message_id = existing_user_message_id
            if session["title"] == "New chat":
                conn.execute(
                    "UPDATE chat_sessions SET title = ? WHERE id = ? AND user_id = ?",
                    (slugish(content), session_id, user["id"]),
                )
            agent_session = _chat_session_for_agent(
                session,
                user["id"],
                model_messages_json=branch["model_messages_json"],
                branch_id=branch["id"],
            )
            uploads = session_upload_context(conn, session_id, user["id"])
            assistant_message_id = insert_assistant_placeholder(
                conn,
                session_id=session_id,
                user_id=user["id"],
                branch_id=branch["id"],
                created_at=now,
            )

        persisted_trace_events: list[dict[str, Any]] = []

        def persist_assistant_message(status: str, text: str | None = None, turn_logs: list[dict[str, Any]] | None = None) -> None:
            metadata = message_metadata(
                status=status,
                trace_events=persisted_trace_events,
                turn_logs=turn_logs,
            )
            with db() as update_conn:
                if text is None:
                    update_conn.execute(
                        "UPDATE chat_messages SET metadata_json = ? WHERE id = ? AND user_id = ?",
                        (json_dumps(metadata), assistant_message_id, user["id"]),
                    )
                else:
                    update_conn.execute(
                        "UPDATE chat_messages SET content = ?, metadata_json = ? WHERE id = ? AND user_id = ?",
                        (text, json_dumps(metadata), assistant_message_id, user["id"]),
                    )

        def wrapped_trace_sink(event: dict[str, Any]) -> None:
            compact = compact_trace_events([event])
            if compact:
                persisted_trace_events.append(compact[0])
                persist_assistant_message("running")
            if trace_sink is not None:
                trace_sink(event)

        agent_prompt = prompt_with_session_context(content, uploads)
        try:
            response, model_messages, turn_logs, trace_events = await run_turn(
                agent_prompt,
                agent_session,
                debug=False,
                trace_sink=wrapped_trace_sink,
            )
        except asyncio.CancelledError:
            persist_assistant_message("failed", "Agent run cancelled.")
            raise
        except Exception as exc:
            error_text = f"Agent run failed: {exc}"
            persist_assistant_message("failed", error_text)
            raise

        assistant_text = response.reply
        metadata = {
            "status": "done",
            "turn_logs": compact_turn_logs(turn_logs),
            "trace_events": compact_trace_events(trace_events),
        }
        now = utc_now()
        with db() as conn:
            conn.execute(
                """
                UPDATE chat_messages
                SET content = ?, metadata_json = ?, created_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (assistant_text, json_dumps(metadata), now, assistant_message_id, user["id"]),
            )
            conn.execute(
                """
                UPDATE chat_sessions
                SET model_messages_json = ?, active_branch_id = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (messages_to_json(model_messages), branch["id"], now, session_id, user["id"]),
            )
            conn.execute(
                """
                UPDATE chat_branches
                SET model_messages_json = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ? AND id = ?
                """,
                (messages_to_json(model_messages), now, session_id, user["id"], branch["id"]),
            )
            _audit(conn, user["id"], "chat_message", {"session_id": session_id})

        return {
            "message": {
                "id": assistant_message_id,
                "role": "assistant",
                "content": assistant_text,
                "created_at": now,
                "branch_id": branch["id"],
                "metadata": metadata,
            },
            "user_message_id": user_message_id,
            "session": {
                "id": session_id,
                "title": agent_session.session_title or "New chat",
                "active_branch_id": branch["id"],
                "updated_at": now,
            },
        }


@app.post("/api/chat/sessions/{session_id}/messages")
async def send_message(session_id: str, body: SendMessageRequest, user: sqlite3.Row = Depends(current_user)) -> dict[str, Any]:
    return await _execute_chat_message(session_id, body.content, user)


@app.post("/api/chat/sessions/{session_id}/messages/stream")
async def stream_message(session_id: str, body: SendMessageRequest, user: sqlite3.Row = Depends(current_user)) -> StreamingResponse:
    async def events():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)

        def enqueue(item: dict[str, Any]) -> None:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

        def sink(event: dict[str, Any]) -> None:
            if event.get("kind") == "text_delta" and event.get("label") == "orchestrator":
                enqueue({"type": "text_delta", "content": event.get("content") or ""})
                return
            compact = compact_trace_events([event])
            if compact:
                enqueue({"type": "trace", "event": compact[0]})

        async def run_and_signal() -> None:
            try:
                payload = await _execute_chat_message(session_id, body.content, user, trace_sink=sink)
                await queue.put({"type": "replace", "content": payload["message"]["content"]})
                await queue.put({"type": "done", "data": payload})
            except Exception as exc:
                await queue.put({"type": "error", "error": str(exc)})

        task = asyncio.create_task(run_and_signal())
        try:
            while True:
                item = await queue.get()
                yield f"data: {json_dumps(item)}\n\n"
                if item["type"] in {"done", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(events(), media_type="text/event-stream")


if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")


@app.get("/{full_path:path}")
def serve_frontend(full_path: str) -> FileResponse:
    if full_path and WEB_DIR.exists():
        target = (WEB_DIR / full_path).resolve()
        root = WEB_DIR.resolve()
        if target.is_file() and target.is_relative_to(root):
            return FileResponse(target)
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(index)
