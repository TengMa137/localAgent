import builtins
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry, RunContext
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

    sanitized, invalid, write_targets = fs_agent._sanitize_objective_paths(
        "find and read the /skills/local-fitness-skills file"
    )

    assert invalid == ["/skills/local-fitness-skills"]
    assert write_targets == []
    assert "/skills/local-fitness-skills" not in sanitized
    assert "local fitness skills" in sanitized


def test_fs_task_prompt_includes_file_index_and_write_targets():
    from agents import fs_agent

    prompt, invalid, write_targets = fs_agent._fs_task_prompt(
        "summarize the /skills/local-fitness-skills file"
    )
    assert invalid == ["/skills/local-fitness-skills"]
    assert write_targets == []
    assert "Invalid exact path hints" in prompt
    assert "/skills/fitness/diet.md" in prompt
    assert "/skills/fitness/workout.md" in prompt

    prompt, invalid, write_targets = fs_agent._fs_task_prompt(
        "create /skills/fitness/movement_recovery.md."
    )
    assert invalid == []
    assert write_targets == ["/skills/fitness/movement_recovery.md"]
    assert "Valid write target path hints" in prompt
    assert "- /skills/fitness/movement_recovery.md" in prompt
    assert "Skill editing policy hook" in prompt
    assert "Source: /skills/skill_editing.md" in prompt


def test_fs_task_prompt_includes_skill_policy_for_loose_skill_write():
    from agents import fs_agent

    prompt, invalid, write_targets = fs_agent._fs_task_prompt(
        "write a new skill under skills/fitness about recovery movements"
    )

    assert invalid == []
    assert write_targets == []
    assert "Skill editing policy hook" in prompt
    assert "Source: /skills/skill_editing.md" in prompt
    assert "Skill Improvement Guidelines" in prompt


def test_fs_task_prompt_omits_skill_policy_for_non_skill_write():
    from agents import fs_agent

    prompt, invalid, write_targets = fs_agent._fs_task_prompt(
        "write a short local note about recovery"
    )

    assert invalid == []
    assert write_targets == []
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
    with pytest.raises(ModelRetry) as exc:
        await orchestrator_agent._run_specialist_once(
            ctx,
            tool_name="run_fs_task",
            objective="  READ   file ",
            runner=lambda objective: _counting_runner(calls, objective),
        )

    assert calls == ["read file"]
    assert "done: read file" in first
    assert "Duplicate specialist call blocked" in str(exc.value)
    assert "done: read file" in str(exc.value)


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
