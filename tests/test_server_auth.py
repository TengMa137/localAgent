import importlib

from fastapi.testclient import TestClient


def _load_server(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOCALAGENT_DB_PATH", str(tmp_path / "state" / "test.sqlite3"))
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

        response = client.post("/api/auth/logout", headers={"X-CSRF-Token": _csrf(client)})
        assert response.status_code == 200
        assert client.get("/api/me").status_code == 401


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
        assert client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 200
        csrf = _csrf(client)
        response = client.patch(
            "/api/me/password",
            headers={"X-CSRF-Token": csrf},
            json={"current_password": "admin-password", "new_password": "new-admin-password"},
        )
        assert response.status_code == 200
        assert client.get("/api/me").status_code == 401

        assert client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 401
        assert client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "new-admin-password"},
        ).status_code == 200


def test_admin_can_reset_user_password(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as admin_client:
        assert admin_client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 200
        csrf = _csrf(admin_client)
        create_user = admin_client.post(
            "/api/admin/users",
            headers={"X-CSRF-Token": csrf},
            json={"username": "normal", "password": "normal-password", "role": "user"},
        )
        assert create_user.status_code == 200
        user_id = create_user.json()["user"]["id"]

        with TestClient(server.app) as user_client:
            assert user_client.post(
                "/api/auth/login",
                json={"username": "normal", "password": "normal-password"},
            ).status_code == 200
            assert user_client.get("/api/me").status_code == 200

            response = admin_client.patch(
                f"/api/admin/users/{user_id}",
                headers={"X-CSRF-Token": csrf},
                json={"password": "changed-password"},
            )
            assert response.status_code == 200
            assert user_client.get("/api/me").status_code == 401
            assert user_client.post(
                "/api/auth/login",
                json={"username": "normal", "password": "normal-password"},
            ).status_code == 401
            assert user_client.post(
                "/api/auth/login",
                json={"username": "normal", "password": "changed-password"},
            ).status_code == 200


def test_blank_normalized_fields_are_rejected(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, tmp_path)
    with TestClient(server.app) as client:
        assert client.post(
            "/api/auth/register",
            json={"username": "   ", "password": "normal-password"},
        ).status_code == 422

        assert client.post(
            "/api/auth/login",
            json={"username": " admin ", "password": "admin-password"},
        ).status_code == 200
        csrf = _csrf(client)

        assert client.post(
            "/api/admin/users",
            headers={"X-CSRF-Token": csrf},
            json={"username": "   ", "password": "normal-password", "role": "user"},
        ).status_code == 422

        create_chat = client.post(
            "/api/chat/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "   "},
        )
        assert create_chat.status_code == 200
        assert create_chat.json()["session"]["title"] == "New chat"
        chat_id = create_chat.json()["session"]["id"]

        assert client.patch(
            f"/api/chat/sessions/{chat_id}",
            headers={"X-CSRF-Token": csrf},
            json={"title": "   "},
        ).status_code == 422
        assert client.post(
            f"/api/chat/sessions/{chat_id}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"content": "   "},
        ).status_code == 422


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
        assert admin_client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 200
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
            assert user_client.post(
                "/api/auth/login",
                json={"username": "normal", "password": "normal-password"},
            ).status_code == 200
            user_csrf = _csrf(user_client)
            assert user_client.get(f"/api/chat/sessions/{admin_chat_id}").status_code == 404
            assert user_client.patch(
                f"/api/chat/sessions/{admin_chat_id}",
                headers={"X-CSRF-Token": user_csrf},
                json={"title": "stolen"},
            ).status_code == 404


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
        assert admin_client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 200

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
        assert client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 200
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
        assert [variant["active"] for variant in edited["branch_variants"]] == [False, True]

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
        assert [variant["active"] for variant in original["branch_variants"]] == [True, False]


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
        assert client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 200
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
        assert client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin-password"},
        ).status_code == 200
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
