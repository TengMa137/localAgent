import builtins
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from tools.filesystem import FilesystemValidator, FilesystemValidatorConfig, Mount


def test_filesystem_write_approval_follows_mount_policy(monkeypatch, tmp_path):
    from agents.runtime import context

    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[
                Mount(
                    host_path=tmp_path / "approved",
                    mount_point="/approved",
                    mode="rw",
                    write_approval=True,
                ),
                Mount(
                    host_path=tmp_path / "quiet",
                    mount_point="/quiet",
                    mode="rw",
                    write_approval=False,
                ),
            ]
        )
    )
    monkeypatch.setattr(context, "validator", validator)

    assert context._fs_needs_approval(
        None,
        SimpleNamespace(name="write_file"),
        {"path": "/approved/a.txt"},
    )
    assert not context._fs_needs_approval(
        None,
        SimpleNamespace(name="write_file"),
        {"path": "/quiet/a.txt"},
    )


def test_filesystem_approval_uses_copy_destination(monkeypatch, tmp_path):
    from agents.runtime import context

    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[
                Mount(
                    host_path=tmp_path / "src",
                    mount_point="/src",
                    mode="rw",
                    write_approval=False,
                ),
                Mount(
                    host_path=tmp_path / "dst",
                    mount_point="/dst",
                    mode="rw",
                    write_approval=True,
                ),
            ]
        )
    )
    monkeypatch.setattr(context, "validator", validator)

    assert context._fs_needs_approval(
        None,
        SimpleNamespace(name="copy_file"),
        {"source": "/src/a.txt", "destination": "/dst/a.txt"},
    )


def test_default_skills_mount_is_writable_with_approval():
    from agents.runtime import context

    assert context.validator.can_write("/skills/new-skill.md")
    assert context._fs_needs_approval(
        None,
        SimpleNamespace(name="write_file"),
        {"path": "/skills/new-skill.md"},
    )


def test_fs_preflight_replaces_missing_path_hint():
    from agents import fs_agent

    sanitized, analysis = fs_agent.PathPreflight(fs_agent._readable_file_index()).analyze(
        "find and read the /skills/local-fitness-skills file"
    )

    assert analysis.invalid_paths == ["/skills/local-fitness-skills"]
    assert analysis.write_targets == []
    assert "/skills/local-fitness-skills" not in sanitized
    assert "local fitness skills" in sanitized


def test_fs_preflight_ignores_urls_before_path_normalization():
    from agents import fs_agent

    sanitized, analysis = fs_agent.PathPreflight([]).analyze(
        "summarize https://example.com/foo.txt and /docs/missing.md"
    )

    assert "https://example.com/foo.txt" in sanitized
    assert "/https://example.com/foo.txt" not in analysis.invalid_paths
    assert analysis.invalid_paths == ["/docs/missing.md"]


def test_fs_preflight_keeps_unmatched_known_suffix_filename():
    from agents import fs_agent

    sanitized, analysis = fs_agent.PathPreflight([]).analyze("read missing_config.yaml")

    assert analysis.invalid_paths == ["/missing_config.yaml"]
    assert "missing config yaml" in sanitized


def test_fs_task_prompt_includes_file_index_and_write_targets():
    from agents import fs_agent

    prompt, analysis = fs_agent._fs_task_prompt(
        "summarize the /skills/local-fitness-skills file"
    )
    assert analysis.invalid_paths == ["/skills/local-fitness-skills"]
    assert analysis.write_targets == []
    assert "Invalid exact path hints" in prompt
    assert "/skills/fitness/diet.md" in prompt
    assert "/skills/fitness/workout.md" in prompt

    prompt, analysis = fs_agent._fs_task_prompt(
        "create /skills/fitness/movement_recovery.md."
    )
    assert analysis.invalid_paths == []
    assert analysis.write_targets == ["/skills/fitness/movement_recovery.md"]
    assert "Valid write target path hints" in prompt
    assert "- /skills/fitness/movement_recovery.md" in prompt
    assert "Skill editing policy hook" in prompt
    assert "Source: /skills/skill_editing.md" in prompt


def test_fs_task_prompt_includes_skill_policy_for_loose_skill_write():
    from agents import fs_agent

    prompt, analysis = fs_agent._fs_task_prompt(
        "write a new skill under skills/fitness about recovery movements"
    )

    assert analysis.invalid_paths == []
    assert analysis.write_targets == []
    assert "Skill editing policy hook" in prompt
    assert "Source: /skills/skill_editing.md" in prompt
    assert "Skill Improvement Guidelines" in prompt


def test_fs_task_prompt_omits_skill_policy_for_non_skill_write():
    from agents import fs_agent

    prompt, analysis = fs_agent._fs_task_prompt(
        "write a short local note about recovery"
    )

    assert analysis.invalid_paths == []
    assert analysis.write_targets == []
    assert "Skill editing policy hook" not in prompt


def test_prompt_for_tool_approval_suggestion(monkeypatch):
    from agents import observability

    replies = iter(["s", "write it under /skills/recovery.md instead"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(replies))
    monkeypatch.delenv("LOCALAGENT_APPROVE_TOOLS", raising=False)

    decision = observability._prompt_for_tool_approval("write_file", "{}")

    assert decision.action == "suggest"
    assert decision.message == "write it under /skills/recovery.md instead"


def test_prompt_for_tool_approval_abort(monkeypatch):
    from agents import observability

    replies = iter(["a", "wrong path"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(replies))
    monkeypatch.delenv("LOCALAGENT_APPROVE_TOOLS", raising=False)

    decision = observability._prompt_for_tool_approval("write_file", "{}")

    assert decision.action == "abort"
    assert decision.message == "wrong path"


def test_prompt_for_tool_approval_non_interactive_denies(monkeypatch):
    from agents import observability

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("LOCALAGENT_APPROVE_TOOLS", raising=False)

    decision = observability._prompt_for_tool_approval("write_file", "{}")

    assert decision.action == "deny"
    assert "stdin is not interactive" in decision.message


def test_write_and_load_agent_report(tmp_path):
    from agents.runtime.reports import load_agent_reports, set_report_dir, write_agent_report

    set_report_dir(tmp_path)
    write_agent_report(
        "fs",
        objective="Read notes",
        summary="Read the requested note.",
        answer="The note says the important fact.",
        findings=["Important fact"],
        paths=["/docs/notes.md"],
        uncertainties=["No open issues"],
    )

    loaded = load_agent_reports(tmp_path)
    assert "REPORT FILE: fs-report.md" in loaded
    assert "Answer:" in loaded
    assert "The note says the important fact." in loaded
    assert "Objective: Read notes" in loaded
    assert "- /docs/notes.md" in loaded

    set_report_dir(None)


def test_agent_report_appends_runs_in_same_session(tmp_path):
    from agents.runtime.reports import load_agent_reports, set_report_dir, write_agent_report

    set_report_dir(tmp_path)
    write_agent_report(
        "fs",
        objective="first",
        summary="first summary",
        answer="first answer",
    )
    write_agent_report(
        "fs",
        objective="second",
        summary="second summary",
        answer="second answer",
    )

    loaded = load_agent_reports(tmp_path)
    assert loaded.count("## Run ") == 2
    assert "Objective: first" in loaded
    assert "Objective: second" in loaded
    assert "first answer" in loaded
    assert "second answer" in loaded

    set_report_dir(None)


def test_fs_success_response_keeps_full_findings_out_of_tool_return():
    from agents.fs_agent import FsAgentResult, _format_success_response

    result = FsAgentResult(
        answer="Use this answer directly.",
        summary="Short filesystem summary.",
        paths=["/docs/a.md"],
        findings=["long finding that should stay in the report"],
        changes_made=["updated /docs/a.md"],
    )

    formatted = _format_success_response(result)

    assert "Forwardable answer:" in formatted
    assert "Use this answer directly." in formatted
    assert "Summary: Short filesystem summary." in formatted
    assert "Detailed findings in fs-report.md: 1 item(s)" in formatted
    assert "long finding that should stay in the report" not in formatted
    assert "updated /docs/a.md" not in formatted


def test_web_response_keeps_full_findings_out_of_tool_return():
    from agents.web_agent import WebAgentResult, _format_orchestrator_response

    result = WebAgentResult(
        answer="Current answer for the user.",
        summary="Short web summary.",
        search_queries=["example query"],
        urls=["https://example.com"],
        findings=["large crawled/RAG finding that belongs in the report"],
    )

    formatted = _format_orchestrator_response(result)

    assert "Forwardable answer:" in formatted
    assert "Current answer for the user." in formatted
    assert "Summary: Short web summary." in formatted
    assert "Detailed findings in web-report.md: 1 item(s)" in formatted
    assert "large crawled/RAG finding that belongs in the report" not in formatted


def test_save_history_includes_report_dir(tmp_path):
    from run_agents import ChatSession, _save_history

    history_path = tmp_path / "chat.json"
    report_dir = tmp_path / "reports" / "session"
    session = ChatSession(
        message_history=[ModelRequest.user_text_prompt("hello")],
        session_title="session",
        history_path=history_path,
        report_dir=report_dir,
    )

    _save_history(session)

    payload = json.loads(history_path.read_text())
    assert payload["session_title"] == "session"
    assert payload["report_dir"] == str(report_dir)
    assert payload["messages"]


def test_new_session_clears_existing_report_dir(monkeypatch, tmp_path):
    import run_agents
    from run_agents import ChatSession, _init_session_paths_from_user_text

    monkeypatch.setattr(run_agents, "CHAT_HISTORY_DIR", tmp_path / "chats")
    monkeypatch.setattr(run_agents, "REPORT_ROOT", tmp_path / "reports")
    stale_report = tmp_path / "reports" / "read-notes" / "fs-report.md"
    stale_report.parent.mkdir(parents=True)
    stale_report.write_text("stale report from an old process", encoding="utf-8")

    session = ChatSession()
    _init_session_paths_from_user_text(session, "read notes")

    assert session.session_title == "read-notes"
    assert session.report_dir == tmp_path / "reports" / "read-notes"
    assert session.report_dir.exists()
    assert not stale_report.exists()


def test_new_session_report_dir_uses_unique_history_stem(monkeypatch, tmp_path):
    import run_agents
    from run_agents import ChatSession, _init_session_paths_from_user_text

    monkeypatch.setattr(run_agents, "CHAT_HISTORY_DIR", tmp_path / "chats")
    monkeypatch.setattr(run_agents, "REPORT_ROOT", tmp_path / "reports")
    existing_history = tmp_path / "chats" / "read-notes.json"
    existing_history.parent.mkdir(parents=True)
    existing_history.write_text("{}", encoding="utf-8")

    session = ChatSession()
    _init_session_paths_from_user_text(session, "read notes")

    assert session.session_title == "read-notes-2"
    assert session.history_path == tmp_path / "chats" / "read-notes-2.json"
    assert session.report_dir == tmp_path / "reports" / "read-notes-2"


def test_skills_context_includes_current_skill_paths():
    from agents.runtime.skills_context import scan_skills_context

    context = scan_skills_context()

    assert "Current /skills catalog" in context
    assert "fitness/diet.md" in context
    assert "fitness/workout.md" in context


def _ctx(run_id: str) -> RunContext:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        run_id=run_id,
    )


async def _counting_runner(calls: list[str], objective: str) -> str:
    calls.append(objective)
    return f"done: {objective}"


@pytest.mark.asyncio
async def test_specialist_guard_bans_same_tool_same_objective():
    from agents import orchestrator_agent

    ctx = _ctx("test-specialist-guard")
    calls: list[str] = []
    orchestrator_agent._tool_run_cache.pop(ctx.run_id, None)

    first = await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="read file",
        runner=lambda objective: _counting_runner(calls, objective),
    )
    second = await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="  READ   file ",
        runner=lambda objective: _counting_runner(calls, objective),
    )

    assert calls == ["read file"]
    assert "done: read file" in first
    assert "Duplicate specialist call blocked" in second
    assert "done: read file" in second


@pytest.mark.asyncio
async def test_specialist_guard_allows_different_tools_and_objectives():
    from agents import orchestrator_agent

    ctx = _ctx("test-specialist-allow-distinct")
    calls: list[str] = []
    orchestrator_agent._tool_run_cache.pop(ctx.run_id, None)

    await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="read file",
        runner=lambda objective: _counting_runner(calls, objective),
    )
    await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_web_task",
        objective="search web",
        runner=lambda objective: _counting_runner(calls, objective),
    )
    await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="edit file",
        runner=lambda objective: _counting_runner(calls, objective),
    )

    assert calls == ["read file", "search web", "edit file"]


@pytest.mark.asyncio
async def test_specialist_guard_uses_turn_id_metadata_for_cache_key():
    from agents import orchestrator_agent

    ctx = _ctx("fallback-run-id")
    ctx.metadata = {"turn_id": "explicit-turn-id"}
    orchestrator_agent._tool_run_cache.pop("explicit-turn-id", None)

    await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="read file",
        runner=lambda objective: _counting_runner([], objective),
    )

    assert orchestrator_agent._tool_run_cache["explicit-turn-id"][-1][2] == "done: read file"


@pytest.mark.asyncio
async def test_specialist_guard_returns_failed_duplicate_without_retry():
    from agents import orchestrator_agent

    ctx = _ctx("test-specialist-failed-duplicate")
    orchestrator_agent._tool_run_cache.pop(ctx.run_id, None)

    async def failed_runner(objective: str) -> str:
        return "Filesystem task failed before a grounded result was produced. Error: nope"

    await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="write file",
        runner=failed_runner,
    )
    second = await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="write file",
        runner=failed_runner,
    )

    assert "Duplicate specialist call blocked" in second
    assert "Error: nope" in second


@pytest.mark.asyncio
async def test_specialist_guard_marks_file_access_problem_terminal():
    from agents import orchestrator_agent

    ctx = _ctx("test-specialist-terminal-access")
    orchestrator_agent._tool_run_cache.pop(ctx.run_id, None)

    async def failed_runner(objective: str) -> str:
        return (
            "I could not complete the filesystem request because of a file access problem.\n"
            "No further agent retry can fix this automatically."
        )

    first = await orchestrator_agent._run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective="edit read only file",
        runner=failed_runner,
    )

    assert "Return this result to the user" in first
    assert "do not create another plan" in first


def test_fs_task_prompt_terminal_missing_file_after_full_index(monkeypatch, tmp_path):
    from agents import fs_agent

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "actual.md").write_text("hello")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    _prompt, analysis = fs_agent._fs_task_prompt("read /docs/not-present.md")

    assert analysis.invalid_paths == ["/docs/not-present.md"]
    assert analysis.terminal_issues
    assert "File not found" in analysis.terminal_issues[0].reason


def test_fs_task_prompt_does_not_infer_write_intent(monkeypatch, tmp_path):
    from agents import fs_agent

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "actual.md").write_text("hello")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    _prompt, analysis = fs_agent._fs_task_prompt("update /docs/actual.md")

    assert analysis.invalid_paths == []
    assert analysis.resolved_paths == ["/docs/actual.md"]
    assert analysis.terminal_issues == []
