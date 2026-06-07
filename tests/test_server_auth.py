import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _load_server(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOCALAGENT_DB_PATH", str(tmp_path / "state" / "test.sqlite3"))
    monkeypatch.setenv("LOCALAGENT_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("LOCALAGENT_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LOCALAGENT_ADMIN_PASSWORD", "admin-password")
    monkeypatch.setenv("LOCALAGENT_COOKIE_SECURE", "false")

    import server

    return importlib.reload(server)


def _csrf(client: TestClient) -> str:
    return client.cookies.get("localagent_csrf") or ""


def test_login_me_logout(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        )
        assert response.status_code == 200
        assert response.json()["user"]["role"] == "admin"
        assert client.cookies.get("localagent_session")
        assert _csrf(client)

        response = client.get("/api/me")
        assert response.status_code == 200
        assert response.json()["user"]["username"] == "admin"

        response = client.post(
            "/api/auth/logout", headers={"X-CSRF-Token": _csrf(client)}
        )
        assert response.status_code == 200
        assert client.get("/api/me").status_code == 401


def test_database_schema_has_no_legacy_report_storage(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    with TestClient(server.app):
        with server.db() as conn:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            columns = {
                f"{table}.{row['name']}"
                for table in tables
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }

    assert not {name for name in tables if "report" in name.lower()}
    assert not {name for name in columns if "report" in name.lower()}


def test_register_creates_normal_user_after_admin_is_initialized(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        response = client.post(
            "/api/auth/register",
            json={"username": "normal", "password": "normal-password"},
        )
        assert response.status_code == 200
        assert response.json()["user"]["role"] == "user"
        assert client.get("/api/me").json()["user"]["username"] == "normal"


def test_user_can_change_own_password(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        response = client.patch(
            "/api/me/password",
            headers={"X-CSRF-Token": csrf},
            json={
                "current_password": "admin-password",
                "new_password": "new-admin-password",
            },
        )
        assert response.status_code == 200
        assert client.get("/api/me").status_code == 401

        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "new-admin-password"},
            ).status_code
            == 200
        )


def test_admin_can_reset_user_password(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as admin_client:
        assert (
            admin_client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(admin_client)
        create_user = admin_client.post(
            "/api/admin/users",
            headers={"X-CSRF-Token": csrf},
            json={"username": "normal", "password": "normal-password", "role": "user"},
        )
        assert create_user.status_code == 200
        user_id = create_user.json()["user"]["id"]

        with TestClient(server.app) as user_client:
            assert (
                user_client.post(
                    "/api/auth/login",
                    json={"username": "normal", "password": "normal-password"},
                ).status_code
                == 200
            )
            assert user_client.get("/api/me").status_code == 200

            response = admin_client.patch(
                f"/api/admin/users/{user_id}",
                headers={"X-CSRF-Token": csrf},
                json={"password": "changed-password"},
            )
            assert response.status_code == 200
            assert user_client.get("/api/me").status_code == 401
            assert (
                user_client.post(
                    "/api/auth/login",
                    json={"username": "normal", "password": "normal-password"},
                ).status_code
                == 401
            )
            assert (
                user_client.post(
                    "/api/auth/login",
                    json={"username": "normal", "password": "changed-password"},
                ).status_code
                == 200
            )


def test_blank_normalized_fields_are_rejected(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/register",
                json={"username": "   ", "password": "normal-password"},
            ).status_code
            == 422
        )

        assert (
            client.post(
                "/api/auth/login",
                json={"username": " admin ", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)

        assert (
            client.post(
                "/api/admin/users",
                headers={"X-CSRF-Token": csrf},
                json={"username": "   ", "password": "normal-password", "role": "user"},
            ).status_code
            == 422
        )

        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "   "},
        )
        assert create_chat.status_code == 200
        assert create_chat.json()["session"]["title"] == "New chat"
        chat_id = create_chat.json()["session"]["id"]

        assert (
            client.patch(
                f"/api/chat/sessions/{chat_id}",
                headers={"X-CSRF-Token": csrf},
                json={"title": "   "},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/chat/sessions/{chat_id}/messages",
                headers={"X-CSRF-Token": csrf},
                json={"content": "   "},
            ).status_code
            == 422
        )


def test_create_chat_reuses_empty_new_chat(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)

        first = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "New chat"},
        )
        second = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "New chat"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["session"]["id"] == first.json()["session"]["id"]
        assert second.json()["session"]["is_empty"] is True

        sessions = client.get("/api/chat/sessions").json()["sessions"]
        empty_new_chats = [
            session
            for session in sessions
            if session["title"] == "New chat" and session["is_empty"]
        ]
        assert len(empty_new_chats) == 1


def test_first_message_preserves_generated_chat_title(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    class FakeResponse:
        reply = "ok"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="ok")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "   "},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        response = client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "Explain branch writes"},
        )
        assert response.status_code == 200
        assert response.json()["session"]["title"] == "explain branch writes"

        loaded = client.get(f"/api/chat/sessions/{chat_id}")
        assert loaded.status_code == 200
        assert loaded.json()["session"]["title"] == "explain branch writes"

        next_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "New chat"},
        )
        assert next_chat.status_code == 200
        assert next_chat.json()["session"]["id"] != chat_id
        assert next_chat.json()["session"]["title"] == "New chat"


def test_first_message_can_use_agent_generated_chat_title(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    class FakeResponse:
        reply = "ok"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        session.session_title = "agent-named-chat"
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="ok")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "   "},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        response = client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "How are you?"},
        )
        assert response.status_code == 200
        assert response.json()["session"]["title"] == "agent-named-chat"


def test_agent_history_is_rebuilt_from_visible_branch_messages(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    captured_histories = []

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        captured_histories.append(str(session.message_history))
        reply = f"answer {len(captured_histories)}"
        return (
            SimpleNamespace(reply=reply),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content=reply)]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "history test"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        first = client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "first question"},
        )
        assert first.status_code == 200

        polluted_history = server.messages_to_json(
            [
                server.ModelRequest.user_text_prompt(
                    "Original objective:\ninternal\n\nPlan workflow result:\nhidden"
                )
            ]
        )
        with server.db() as conn:
            conn.execute(
                "UPDATE chat_sessions SET model_messages_json = ? WHERE id = ?",
                (polluted_history, chat_id),
            )
            conn.execute(
                """
                UPDATE chat_branches
                SET model_messages_json = ?
                WHERE session_id = ? AND id = 'main'
                """,
                (polluted_history, chat_id),
            )

        second = client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "second question"},
        )
        assert second.status_code == 200

    assert captured_histories[0] == "[]"
    assert "first question" in captured_histories[1]
    assert "answer 1" in captured_histories[1]
    assert "Original objective" not in captured_histories[1]
    assert "Plan workflow result" not in captured_histories[1]


def test_stream_does_not_forward_hidden_orchestrator_text_delta(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    class FakeResponse:
        reply = "final answer"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        if trace_sink is not None:
            trace_sink(
                {
                    "kind": "text_delta",
                    "label": "orchestrator",
                    "content": "hidden route decision text",
                }
            )
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="final answer")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "stream test"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        with client.stream(
            "POST",
            f"/api/chat/sessions/{chat_id}/messages/stream",
            headers={"X-CSRF-Token": csrf},
            json={"content": "plan question"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    assert "text_delta" in body
    assert "hidden route decision text" not in body
    assert "final answer" in body


def test_stream_forwards_synthesis_text_delta(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    class FakeResponse:
        reply = "streamed final answer"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        if trace_sink is not None:
            trace_sink(
                {
                    "kind": "text_delta",
                    "label": "synthesis",
                    "content": "streamed ",
                }
            )
            trace_sink(
                {
                    "kind": "text_delta",
                    "label": "synthesis",
                    "content": "final answer",
                }
            )
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(
                    parts=[server.TextPart(content="streamed final answer")]
                ),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "stream synthesis test"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        with client.stream(
            "POST",
            f"/api/chat/sessions/{chat_id}/messages/stream",
            headers={"X-CSRF-Token": csrf},
            json={"content": "plan question"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    assert "streamed " in body
    assert "final answer" in body
    assert "replace" in body


def test_trace_event_compaction_preserves_all_tool_activity(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    events = [
        {
            "kind": "tool_call",
            "label": "fs_agent",
            "tool_name": "read_file",
            "tool_call_id": str(index),
            "args": f'{{"path": "/docs/{index}.md"}}',
        }
        for index in range(120)
    ]
    events.extend(
        [
            {"kind": "text_delta", "label": "orchestrator", "content": "answer"},
            {"kind": "tool_args_delta", "label": "fs_agent", "args_delta": "x"},
        ]
    )

    compact = server.compact_trace_events(events)

    assert len(compact) == 120
    assert {event["kind"] for event in compact} == {"tool_call"}
    assert compact[0]["tool_call_id"] == "0"
    assert compact[-1]["tool_call_id"] == "119"


def test_trace_event_compaction_caps_pathological_activity(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    events = [
        {
            "kind": "tool_result",
            "label": "fs_agent",
            "tool_name": "read_file",
            "tool_call_id": str(index),
            "output": "x" * 5000,
        }
        for index in range(600)
    ]

    compact = server.compact_trace_events(events)

    assert len(compact) <= 500
    assert len(compact[0]["output"]) < 2100


def test_user_can_transcribe_voice_input(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    captured = {}

    class FakeASRProvider:
        async def transcribe_bytes(
            self, audio_bytes, *, filename, mime_type, language=None
        ):
            captured["audio_bytes"] = audio_bytes
            captured["filename"] = filename
            captured["mime_type"] = mime_type
            captured["language"] = language
            return SimpleNamespace(
                text="hello from voice", language="en", provider="fake-asr"
            )

    monkeypatch.setattr(server, "Qwen3ASRProvider", FakeASRProvider)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        response = client.post(
            "/api/speech/asr",
            headers={"X-CSRF-Token": _csrf(client)},
            files={"file": ("recording.webm", b"webm bytes", "audio/webm")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "hello from voice",
        "language": "en",
        "provider": "fake-asr",
    }
    assert captured == {
        "audio_bytes": b"webm bytes",
        "filename": "recording.webm",
        "mime_type": "audio/webm",
        "language": "English",
    }


def test_voice_input_uses_provider_language_when_client_omits_field(
    monkeypatch, tmp_path
):
    server = _load_server(monkeypatch, tmp_path)
    captured = {}

    class FakeASRProvider:
        config = SimpleNamespace(language="Swedish")

        async def transcribe_bytes(
            self, audio_bytes, *, filename, mime_type, language=None
        ):
            captured["language"] = language
            return SimpleNamespace(text="hej", language="sv", provider="fake-asr")

    monkeypatch.setattr(server, "Qwen3ASRProvider", FakeASRProvider)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        response = client.post(
            "/api/speech/asr",
            headers={"X-CSRF-Token": _csrf(client)},
            files={"file": ("recording.webm", b"webm bytes", "audio/webm")},
        )

    assert response.status_code == 200
    assert captured["language"] == "Swedish"


def test_image_upload_context_tells_agent_to_use_read_image(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    captured = {}

    class FakeResponse:
        reply = "ok"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        captured["prompt"] = user_text
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="ok")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "image context"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        upload = client.post(
            f"/api/chat/sessions/{chat_id}/files",
            headers={"X-CSRF-Token": csrf},
            files={"file": ("screenshot.png", b"\x89PNG\r\n\x1a\nfake", "image/png")},
        )
        assert upload.status_code == 200

        response = client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "describe the screenshot"},
        )
        assert response.status_code == 200

    assert "screenshot.png [image, image/png" in captured["prompt"]
    assert "(image; use read_image to inspect visual content)" in captured["prompt"]
    assert (
        "Supported PNG/JPEG/GIF/WebP uploads can be inspected with read_image."
        in captured["prompt"]
    )
    assert "call read_image for supported image paths" in captured["prompt"]


def test_pdf_upload_context_tells_agent_to_use_rag(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    captured = {}

    class FakeResponse:
        reply = "ok"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        captured["prompt"] = user_text
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="ok")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "pdf context"},
        )
        chat_id = create_chat.json()["session"]["id"]

        upload = client.post(
            f"/api/chat/sessions/{chat_id}/files",
            headers={"X-CSRF-Token": csrf},
            files={"file": ("paper.pdf", b"%PDF-1.4\nfixture", "application/pdf")},
        )
        assert upload.status_code == 200

        response = client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "summarize the uploaded paper"},
        )
        assert response.status_code == 200

    assert "paper.pdf [document, application/pdf" in captured["prompt"]
    assert "PDF document; use filesystem/RAG retrieval" in captured["prompt"]
    assert "PDF uploads are document files and must be inspected through RAG" in captured[
        "prompt"
    ]
    assert "not read_file/read_lines/grep_files" in captured["prompt"]


def test_delete_chat_session_removes_upload_directory(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "upload cleanup"},
        )
        chat_id = create_chat.json()["session"]["id"]
        upload = client.post(
            f"/api/chat/sessions/{chat_id}/files",
            headers={"X-CSRF-Token": csrf},
            files={"file": ("notes.txt", b"temporary", "text/plain")},
        )
        assert upload.status_code == 200

        upload_dir = server.WEB_UPLOAD_ROOT / "1" / chat_id
        assert upload_dir.is_dir()

        response = client.delete(
            f"/api/chat/sessions/{chat_id}",
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200
        assert not upload_dir.exists()
        assert not upload_dir.parent.exists()


def test_web_agent_session_uses_per_user_memory_dir(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": _csrf(client)},
            json={"title": "memory path"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

    with server.db() as conn:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (chat_id,)
        ).fetchone()

    agent_session = server._chat_session_for_agent(row, 1)

    assert agent_session.memory_dir == server.STATE_DIR / "memory" / "1"


def test_unsupported_image_upload_context_is_not_routed_to_read_image(
    monkeypatch, tmp_path
):
    server = _load_server(monkeypatch, tmp_path)
    captured = {}

    class FakeResponse:
        reply = "ok"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        captured["prompt"] = user_text
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="ok")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "unsupported image context"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        upload = client.post(
            f"/api/chat/sessions/{chat_id}/files",
            headers={"X-CSRF-Token": csrf},
            files={"file": ("photo.heic", b"heic bytes", "image/heic")},
        )
        assert upload.status_code == 200

        response = client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "describe the uploaded photo"},
        )
        assert response.status_code == 200

    assert "photo.heic [binary, image/heic" in captured["prompt"]
    assert "(unsupported image for read_image; stored binary)" in captured["prompt"]
    assert "photo.heic [image, image/heic" not in captured["prompt"]


def test_register_is_blocked_until_backend_admin_is_initialized(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOCALAGENT_DB_PATH", str(tmp_path / "state" / "test.sqlite3"))
    monkeypatch.setenv("LOCALAGENT_ADMIN_USERNAME", "")
    monkeypatch.setenv("LOCALAGENT_ADMIN_PASSWORD", "")
    monkeypatch.setenv("LOCALAGENT_COOKIE_SECURE", "false")

    import server

    server = importlib.reload(server)

    with TestClient(server.app) as client:
        response = client.post(
            "/api/auth/register",
            json={"username": "normal", "password": "normal-password"},
        )
        assert response.status_code == 409
        assert "Admin account is not initialized" in response.json()["detail"]


def test_users_cannot_read_each_others_chat_sessions(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as admin_client:
        assert (
            admin_client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        admin_csrf = _csrf(admin_client)
        create_user = admin_client.post(
            "/api/admin/users",
            headers={"X-CSRF-Token": admin_csrf},
            json={"username": "normal", "password": "normal-password", "role": "user"},
        )
        assert create_user.status_code == 200
        create_chat = admin_client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": admin_csrf},
            json={"title": "admin private chat"},
        )
        assert create_chat.status_code == 200
        admin_chat_id = create_chat.json()["session"]["id"]

        with TestClient(server.app) as user_client:
            assert (
                user_client.post(
                    "/api/auth/login",
                    json={"username": "normal", "password": "normal-password"},
                ).status_code
                == 200
            )
            user_csrf = _csrf(user_client)
            assert (
                user_client.get(f"/api/chat/sessions/{admin_chat_id}").status_code
                == 404
            )
            assert (
                user_client.patch(
                    f"/api/chat/sessions/{admin_chat_id}",
                    headers={"X-CSRF-Token": user_csrf},
                    json={"title": "stolen"},
                ).status_code
                == 404
            )


def test_admin_can_view_all_user_chat_history(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as user_client:
        register = user_client.post(
            "/api/auth/register",
            json={"username": "normal", "password": "normal-password"},
        )
        assert register.status_code == 200
        user_id = register.json()["user"]["id"]
        user_csrf = _csrf(user_client)
        create_chat = user_client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": user_csrf},
            json={"title": "normal private chat"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        with server.db() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, role, content, created_at)
                VALUES (?, ?, 'user', ?, ?)
                """,
                (chat_id, user_id, "private question", server._utc_now()),
            )

        assert user_client.get("/api/admin/chat/sessions").status_code == 403

    with TestClient(server.app) as admin_client:
        assert (
            admin_client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )

        response = admin_client.get("/api/admin/chat/sessions")
        assert response.status_code == 200
        sessions = response.json()["sessions"]
        assert any(
            session["id"] == chat_id
            and session["username"] == "normal"
            and session["title"] == "normal private chat"
            for session in sessions
        )

        response = admin_client.get(f"/api/admin/chat/sessions/{chat_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session"]["username"] == "normal"
        assert data["messages"][0]["content"] == "private question"


def test_editing_user_message_creates_switchable_branch(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    class FakeResponse:
        reply = "edited answer"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="edited answer")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "branch test"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        with server.db() as conn:
            now = server._utc_now()
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, branch_id, role, content, created_at)
                VALUES (?, 1, 'main', 'user', 'first', ?)
                """,
                (chat_id, now),
            )
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, branch_id, role, content, created_at)
                VALUES (?, 1, 'main', 'assistant', 'first answer', ?)
                """,
                (chat_id, now),
            )
            second = conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, branch_id, role, content, created_at)
                VALUES (?, 1, 'main', 'user', 'second original', ?)
                """,
                (chat_id, now),
            ).lastrowid
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, branch_id, role, content, created_at)
                VALUES (?, 1, 'main', 'assistant', 'second answer', ?)
                """,
                (chat_id, now),
            )

        response = client.post(
            f"/api/chat/sessions/{chat_id}/messages/{second}/fork",
            headers={"X-CSRF-Token": csrf},
            json={"content": "second edited"},
        )
        assert response.status_code == 200
        branch_id = response.json()["session"]["active_branch_id"]
        assert branch_id != "main"

        response = client.get(f"/api/chat/sessions/{chat_id}")
        assert response.status_code == 200
        data = response.json()
        assert [message["content"] for message in data["messages"]] == [
            "first",
            "first answer",
            "second edited",
            "edited answer",
        ]
        edited = data["messages"][2]
        assert [variant["number"] for variant in edited["branch_variants"]] == [1, 2]
        assert [variant["active"] for variant in edited["branch_variants"]] == [
            False,
            True,
        ]

        response = client.post(
            f"/api/chat/sessions/{chat_id}/branches/main/activate",
            headers={"X-CSRF-Token": csrf},
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert [message["content"] for message in data["messages"]] == [
            "first",
            "first answer",
            "second original",
            "second answer",
        ]
        original = data["messages"][2]
        assert [variant["number"] for variant in original["branch_variants"]] == [1, 2]
        assert [variant["active"] for variant in original["branch_variants"]] == [
            True,
            False,
        ]


def test_fork_allows_user_message_from_inactive_branch(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)

    class FakeResponse:
        reply = "forked answer"

    async def fake_run_turn(user_text, session, debug=False, trace_sink=None):
        return (
            FakeResponse(),
            [
                *session.message_history,
                server.ModelRequest.user_text_prompt(user_text),
                server.ModelResponse(parts=[server.TextPart(content="forked answer")]),
            ],
            [],
            [],
        )

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "inactive branch fork"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        with server.db() as conn:
            now = server._utc_now()
            main_message_id = conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, branch_id, role, content, created_at)
                VALUES (?, 1, 'main', 'user', 'original branch message', ?)
                """,
                (chat_id, now),
            ).lastrowid
            conn.execute(
                """
                INSERT INTO chat_branches (
                    session_id, user_id, id, parent_id, fork_parent_message_id,
                    variant_number, model_messages_json, created_at, updated_at
                )
                VALUES (?, 1, 'other', NULL, NULL, 1, '[]', ?, ?)
                """,
                (chat_id, now, now),
            )
            conn.execute(
                """
                UPDATE chat_sessions
                SET active_branch_id = 'other'
                WHERE id = ? AND user_id = 1
                """,
                (chat_id,),
            )

        response = client.post(
            f"/api/chat/sessions/{chat_id}/messages/{main_message_id}/fork",
            headers={"X-CSRF-Token": csrf},
            json={"content": "edited while inactive"},
        )
        assert response.status_code == 200
        assert response.json()["session"]["active_branch_id"] not in {"main", "other"}


def test_stale_running_message_is_marked_failed_on_session_load(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "stale running"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        with server.db() as conn:
            now = server._utc_now()
            conn.execute(
                """
                INSERT INTO chat_messages (
                    session_id, user_id, branch_id, role, content, metadata_json, created_at
                )
                VALUES (?, 1, 'main', 'assistant', '', ?, ?)
                """,
                (chat_id, server._json_dumps({"status": "running"}), now),
            )

        response = client.get(f"/api/chat/sessions/{chat_id}")
        assert response.status_code == 200
        message = response.json()["messages"][0]
        assert message["metadata"]["status"] == "failed"
        assert message["content"] == "Agent run stopped before completing."


def test_chat_session_detail_returns_bounded_message_tail(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "bounded history"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]

        with server.db() as conn:
            now = server._utc_now()
            for index in range(server.MAX_CHAT_MESSAGES_RETURNED + 5):
                conn.execute(
                    """
                    INSERT INTO chat_messages (
                        session_id, user_id, branch_id, role, content, created_at
                    )
                    VALUES (?, 1, 'main', 'user', ?, ?)
                    """,
                    (chat_id, f"message {index}", now),
                )

        response = client.get(f"/api/chat/sessions/{chat_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["messages_truncated"] is True
        assert data["total_messages"] == server.MAX_CHAT_MESSAGES_RETURNED + 5
        assert len(data["messages"]) == server.MAX_CHAT_MESSAGES_RETURNED
        assert data["messages"][0]["content"] == "message 5"


def test_chat_session_detail_truncates_oversized_message_content(
    monkeypatch, tmp_path
):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "admin-password"},
            ).status_code
            == 200
        )
        csrf = _csrf(client)
        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "bounded content"},
        )
        assert create_chat.status_code == 200
        chat_id = create_chat.json()["session"]["id"]
        content = "x" * (server.MAX_PUBLIC_MESSAGE_CHARS + 500)

        with server.db() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    session_id, user_id, branch_id, role, content, created_at
                )
                VALUES (?, 1, 'main', 'assistant', ?, ?)
                """,
                (chat_id, content, server._utc_now()),
            )

        response = client.get(f"/api/chat/sessions/{chat_id}")
        assert response.status_code == 200
        message = response.json()["messages"][0]
        assert message["content_truncated"] is True
        assert message["content_original_length"] == len(content)
        assert len(message["content"]) <= server.MAX_PUBLIC_MESSAGE_CHARS
