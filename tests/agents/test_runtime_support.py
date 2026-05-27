import builtins
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest

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

    sanitized, analysis = fs_agent.PathPreflight(
        fs_agent._readable_file_index()
    ).analyze("find and read the /skills/local-fitness-skills file")

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
    assert "Use the Readable file index below before calling discovery tools." in prompt
    assert "Never invent paths." in prompt
    assert "Never read the same path twice in one run." in prompt

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
    from localagent_settings import get_runtime_settings

    replies = iter(["s", "write it under /skills/recovery.md instead"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(replies))
    monkeypatch.delenv("LOCALAGENT_APPROVE_TOOLS", raising=False)
    get_runtime_settings.cache_clear()

    decision = observability._prompt_for_tool_approval("write_file", "{}")

    assert decision.action == "suggest"
    assert decision.message == "write it under /skills/recovery.md instead"
    get_runtime_settings.cache_clear()


def test_prompt_for_tool_approval_abort(monkeypatch):
    from agents import observability
    from localagent_settings import get_runtime_settings

    replies = iter(["a", "wrong path"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(replies))
    monkeypatch.delenv("LOCALAGENT_APPROVE_TOOLS", raising=False)
    get_runtime_settings.cache_clear()

    decision = observability._prompt_for_tool_approval("write_file", "{}")

    assert decision.action == "abort"
    assert decision.message == "wrong path"
    get_runtime_settings.cache_clear()


def test_prompt_for_tool_approval_non_interactive_denies(monkeypatch):
    from agents import observability
    from localagent_settings import get_runtime_settings

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("LOCALAGENT_APPROVE_TOOLS", raising=False)
    get_runtime_settings.cache_clear()

    decision = observability._prompt_for_tool_approval("write_file", "{}")

    assert decision.action == "deny"
    assert "stdin is not interactive" in decision.message
    get_runtime_settings.cache_clear()


def test_write_and_load_agent_report(tmp_path):
    from agents.runtime.reports import (
        load_agent_reports,
        set_report_dir,
        write_agent_report,
    )

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
    from agents.runtime.reports import (
        load_agent_reports,
        set_report_dir,
        write_agent_report,
    )

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


def test_agent_report_writes_are_serialized(monkeypatch, tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from agents.runtime import reports

    reports.set_report_dir(tmp_path)
    first_write_started = threading.Event()
    delayed_once = False
    original_write_text = Path.write_text

    def delayed_write_text(path, data, *args, **kwargs):
        nonlocal delayed_once
        if (
            path.name == "fs-report.md"
            and "Objective: first" in data
            and not delayed_once
        ):
            delayed_once = True
            first_write_started.set()
            time.sleep(0.2)
        return original_write_text(path, data, *args, **kwargs)

    def write_report(objective: str) -> None:
        reports.set_report_dir(tmp_path)
        reports.write_agent_report(
            "fs",
            objective=objective,
            summary=f"{objective} summary",
            answer=f"{objective} answer",
        )

    monkeypatch.setattr(Path, "write_text", delayed_write_text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(write_report, "first")
        assert first_write_started.wait(timeout=1)
        second = executor.submit(write_report, "second")
        first.result(timeout=2)
        second.result(timeout=2)

    content = (tmp_path / "fs-report.md").read_text(encoding="utf-8")
    assert "Objective: first" in content
    assert "Objective: second" in content
    assert content.count("## Run ") == 2

    reports.set_report_dir(None)


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


def test_web_query_guidance_includes_time_sensitive_semantic_guidance():
    from agents.web_agent import _web_query_guidance

    guidance = _web_query_guidance("What's today's gold price?")

    assert "Current date/time:" in guidance
    assert (
        "Choose the first web_search_tool query semantically from the objective"
        in guidance
    )
    assert "avoid adding a bare year" in guidance


def test_orchestrator_decision_requires_route_payload():
    from pydantic import ValidationError

    from agents.orchestrator_agent import OrchestratorDecision

    with pytest.raises(ValidationError):
        OrchestratorDecision(route="direct")

    with pytest.raises(ValidationError):
        OrchestratorDecision(route="plan")

    assert OrchestratorDecision(route="direct", reply="Hello").reply == "Hello"
    assert (
        OrchestratorDecision(
            route="plan",
            objective="Read the requested file",
            effort="minimal",
        ).objective
        == "Read the requested file"
    )


def test_agent_runtime_settings_read_dotenv(monkeypatch, tmp_path):
    from localagent_settings import AgentRuntimeSettings, get_runtime_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCALAGENT_MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("LOCALAGENT_ORCHESTRATOR_USE_XML", raising=False)
    monkeypatch.delenv("LOCALAGENT_SKILLS_MODE", raising=False)
    assert AgentRuntimeSettings().orchestrator_use_xml is False

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LOCALAGENT_MEMORY_ENABLED=false",
                "LOCALAGENT_ORCHESTRATOR_USE_XML=xml",
                "LOCALAGENT_SKILLS_MODE=RO",
            ]
        ),
        encoding="utf-8",
    )
    get_runtime_settings.cache_clear()

    settings = get_runtime_settings()

    assert settings.memory_enabled is False
    assert settings.orchestrator_use_xml is True
    assert settings.skills_mode == "ro"
    get_runtime_settings.cache_clear()


def test_env_example_contains_only_supported_settings():
    import re
    from pathlib import Path

    from localagent_settings import AgentRuntimeSettings, SpeechSettings

    repo_root = Path(__file__).resolve().parents[2]
    env_text = (repo_root / ".env.example").read_text(encoding="utf-8")
    env_keys = {
        match.group(1)
        for line in env_text.splitlines()
        if (match := re.match(r"#?\s*([A-Z][A-Z0-9_]+)=", line))
    }
    runtime_keys = {
        f"LOCALAGENT_{field_name.upper()}"
        for field_name in AgentRuntimeSettings.model_fields
    }
    speech_keys = {
        f"LOCALAGENT_{field_name.upper()}" for field_name in SpeechSettings.model_fields
    }
    server_keys = {
        "LOCALAGENT_STATE_DIR",
        "LOCALAGENT_DB_PATH",
        "LOCALAGENT_WEB_DIR",
        "LOCALAGENT_DOCS_DIR",
        "LOCALAGENT_MAX_UPLOAD_BYTES",
        "LOCALAGENT_SESSION_COOKIE",
        "LOCALAGENT_COOKIE_SECURE",
        "LOCALAGENT_SESSION_TTL_SECONDS",
        "LOCALAGENT_ADMIN_USERNAME",
        "LOCALAGENT_ADMIN_PASSWORD",
        "LOCALAGENT_MAX_VOICE_AUDIO_BYTES",
    }
    compose_keys = {"LOCALAGENT_BIND", "LOCALAGENT_PORT"}

    supported = runtime_keys | speech_keys | server_keys | compose_keys

    assert "OPENAI_API_KEY" not in env_keys
    assert "ANTHROPIC_API_KEY" not in env_keys
    assert env_keys <= supported
    assert runtime_keys <= env_keys
    assert speech_keys <= env_keys


def test_parse_xml_orchestrator_decision_with_memory():
    from agents.orchestrator_agent import _parse_xml_orchestrator_decision

    decision = _parse_xml_orchestrator_decision(
        """
<decision>
  <route>plan</route>
  <reply></reply>
  <objective>Read /docs/notes.md and summarize it.</objective>
  <effort>minimal</effort>
  <routing_reason>The request needs local file access.</routing_reason>
  <session_title>read-notes</session_title>
  <memory_findings>
    <finding>
      <category>preference</category>
      <text>User prefers concise answers with direct file references.</text>
      <explicit>true</explicit>
      <confidence>0.95</confidence>
      <sensitivity>low</sensitivity>
      <reason>User stated a durable response preference.</reason>
    </finding>
  </memory_findings>
</decision>
"""
    )

    assert decision.route == "plan"
    assert decision.objective == "Read /docs/notes.md and summarize it."
    assert decision.reply is None
    assert decision.effort == "minimal"
    assert decision.session_title == "read-notes"
    assert len(decision.memory_findings) == 1
    assert decision.memory_findings[0].explicit is True
    assert decision.memory_findings[0].confidence == 0.95


def test_parse_xml_orchestrator_decision_accepts_fenced_document():
    from agents.orchestrator_agent import _parse_xml_orchestrator_decision

    decision = _parse_xml_orchestrator_decision(
        """
```xml
<decision>
  <route>direct</route>
  <reply><![CDATA[Use general reasoning for stable concepts.]]></reply>
  <objective></objective>
  <effort>none</effort>
  <routing_reason><![CDATA[No retrieval is needed.]]></routing_reason>
  <session_title></session_title>
  <memory_findings/>
</decision>
```
"""
    )

    assert decision.route == "direct"
    assert decision.reply == "Use general reasoning for stable concepts."


@pytest.mark.asyncio
async def test_orchestrator_turn_uses_xml_output_contract(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import run_orchestrator_turn
    from localagent_settings import get_runtime_settings

    seen: dict[str, object] = {}

    async def fake_observable_run(agent, prompt, **_kwargs):
        seen["agent"] = agent
        seen["prompt"] = prompt
        return SimpleNamespace(
            output="""
<decision>
  <route>direct</route>
  <reply>Hello.</reply>
  <objective></objective>
  <effort>none</effort>
  <routing_reason>Greeting can be answered directly.</routing_reason>
  <session_title>hello</session_title>
  <memory_findings/>
</decision>
""",
            all_messages=lambda: [ModelRequest.user_text_prompt(prompt)],
        )

    monkeypatch.setenv("LOCALAGENT_ORCHESTRATOR_USE_XML", "true")
    get_runtime_settings.cache_clear()
    monkeypatch.setattr(orchestrator_agent, "observable_run", fake_observable_run)

    result = await run_orchestrator_turn("hello")

    assert seen["agent"] is orchestrator_agent.orchestrator_xml
    assert "XML output format:" in orchestrator_agent._orchestrator_xml_prompt()
    assert result.output.reply == "Hello."
    assert result.decision.route == "direct"
    assert result.decision.session_title == "hello"
    get_runtime_settings.cache_clear()


@pytest.mark.asyncio
async def test_orchestrator_xml_output_repair_retry(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import run_orchestrator_turn

    calls: list[str] = []

    async def fake_observable_run(_agent, prompt, **_kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            output = "not xml"
        else:
            output = """
<decision>
  <route>direct</route>
  <reply>Repaired.</reply>
  <objective></objective>
  <effort>none</effort>
  <routing_reason>Direct answer after repair.</routing_reason>
  <session_title></session_title>
  <memory_findings/>
</decision>
"""
        return SimpleNamespace(
            output=output,
            all_messages=lambda: [ModelRequest.user_text_prompt(prompt)],
        )

    monkeypatch.setattr(orchestrator_agent, "observable_run", fake_observable_run)

    result = await run_orchestrator_turn("hello", use_xml=True)

    assert len(calls) == 2
    assert "failed XML parsing or schema validation" in calls[1]
    assert "Original user prompt to route:\nhello" in calls[1]
    assert result.output.reply == "Repaired."


@pytest.mark.asyncio
async def test_orchestrator_xml_retries_schema_invalid_payload(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import _run_xml_orchestrator_decision

    calls: list[str] = []

    async def fake_observable_run(_agent, prompt, **_kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            output = "not xml"
        elif len(calls) == 2:
            output = """
<decision>
  <route>direct</route>
  <reply></reply>
  <objective><![CDATA[Read all available fitness skills.]]></objective>
  <effort>minimal</effort>
  <routing_reason><![CDATA[Invalid direct payload with objective.]]></routing_reason>
  <session_title>available-fitness-skills</session_title>
  <memory_findings/>
</decision>
"""
        else:
            output = """
<decision>
  <route>plan</route>
  <reply></reply>
  <objective><![CDATA[Read all fitness skills and list all important points.]]></objective>
  <effort>minimal</effort>
  <routing_reason><![CDATA[The request needs filesystem retrieval.]]></routing_reason>
  <session_title>fitness-skills-summary</session_title>
  <memory_findings/>
</decision>
"""
        return SimpleNamespace(
            output=output,
            all_messages=lambda: [ModelRequest.user_text_prompt(prompt)],
        )

    monkeypatch.setattr(orchestrator_agent, "observable_run", fake_observable_run)

    result = await _run_xml_orchestrator_decision(
        "read all fitness skills and list all important points",
        label="orchestrator",
        indent=0,
        message_history=[],
        metadata=None,
    )

    assert len(calls) == 3
    assert "reply is required for direct" in calls[2]
    assert "route=plan requires a non-empty <objective>" in calls[2]
    assert result.output.route == "plan"
    assert result.output.objective == (
        "Read all fitness skills and list all important points."
    )


def test_orchestrator_prompt_declares_validator_mount_access():
    from agents.orchestrator_agent import _orchestrator_prompt_body

    prompt = _orchestrator_prompt_body()

    assert "Filesystem access contract:" in prompt
    assert "/docs" in prompt
    assert "/skills" in prompt
    assert "choose plan" in prompt
    assert "Do not answer that you lack access" in prompt


def test_orchestrator_prompt_uses_only_direct_and_plan_routes():
    from agents.orchestrator_agent import _orchestrator_prompt_body

    prompt = _orchestrator_prompt_body()
    normalized = " ".join(prompt.split())

    assert "Prefer making progress:" in prompt
    assert "Choose direct when" in prompt
    assert "Choose plan when" in prompt
    assert "There is no clarify route" in prompt
    assert "Classify by the information source and missing user intent" in prompt
    assert "Stable conceptual" in prompt
    assert "not by topic keywords" in normalized
    assert "not executing the route" in prompt
    assert "machine learning" not in prompt.lower()
    assert "direct|clarify" not in prompt


@pytest.mark.asyncio
async def test_orchestrator_direct_decision_returns_reply():
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages

    response, messages = await _response_and_messages(
        OrchestratorDecision(route="direct", reply="Use the cached answer."),
        [],
    )

    assert response.reply == "Use the cached answer."
    assert messages == []


@pytest.mark.asyncio
async def test_orchestrator_plan_decision_uses_effort_budget(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        OrchestratorResponse,
        _response_and_messages,
    )

    calls: list[tuple[str, int, int]] = []
    answer_prompts: list[str] = []

    async def fake_plan_workflow(
        objective: str,
        *,
        max_tasks: int,
        max_iterations: int,
    ) -> str:
        calls.append((objective, max_tasks, max_iterations))
        return (
            "Forwardable answer:\n"
            f"Read result for {objective}.\n\n"
            "Orchestrator notes:\n- Detailed findings in plan-report.md: 1 item(s)"
        )

    async def fake_observable_run(_agent, prompt, **_kwargs):
        answer_prompts.append(prompt)
        return SimpleNamespace(
            output=OrchestratorResponse(reply="Final answer."),
            all_messages=lambda: [],
        )

    monkeypatch.setattr(orchestrator_agent, "_run_plan_workflow", fake_plan_workflow)
    monkeypatch.setattr(orchestrator_agent, "observable_run", fake_observable_run)
    monkeypatch.setattr(
        orchestrator_agent,
        "_load_agent_report_summaries",
        lambda _report_dir: "REPORT FILE: fs-report.md\nSummary:\nRead notes.",
    )
    monkeypatch.setattr(orchestrator_agent, "_current_report_dir", lambda: None)

    response, messages = await _response_and_messages(
        OrchestratorDecision(
            route="plan",
            objective="read notes.md",
            effort="minimal",
            session_title="read-notes",
        ),
        [],
    )

    assert response.reply == "Final answer."
    assert messages == []
    assert response.session_title == "read-notes"
    assert calls == [("read notes.md", 1, 1)]
    assert "Suggested session title:\nread-notes" in answer_prompts[0]
    assert "Forwardable answer:" in answer_prompts[0]
    assert "Orchestrator notes:" in answer_prompts[0]
    assert "Compact session report summaries" in answer_prompts[0]
    assert "REPORT FILE: fs-report.md" in answer_prompts[0]


@pytest.mark.asyncio
async def test_orchestrator_plan_answer_falls_back_to_forwardable_research(
    monkeypatch,
):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        OrchestratorResponse,
        _response_and_messages,
    )

    forwardable = (
        "Gold price research found a current web result: Rs 15,888 per gram "
        "for 24K gold, with uncertainty around whether it is a live quote."
    )

    async def fake_plan_workflow(
        _objective: str,
        *,
        max_tasks: int,
        max_iterations: int,
    ) -> str:
        return (
            "Forwardable answer:\n"
            f"{forwardable}\n\n"
            "Orchestrator notes:\n"
            "- Execution status: complete-with-uncertainties\n"
            "- Sources: https://www.goodreturns.in/gold-rates/"
        )

    async def fake_observable_run(_agent, _prompt, **_kwargs):
        return SimpleNamespace(
            output=OrchestratorResponse(
                reply=(
                    "I'm sorry, but I don't have access to real-time financial "
                    "data. Please check a market data website."
                )
            ),
            all_messages=lambda: [],
        )

    monkeypatch.setattr(orchestrator_agent, "_run_plan_workflow", fake_plan_workflow)
    monkeypatch.setattr(orchestrator_agent, "observable_run", fake_observable_run)
    monkeypatch.setattr(orchestrator_agent, "_current_report_dir", lambda: None)

    response, messages = await _response_and_messages(
        OrchestratorDecision(
            route="plan",
            objective="Research current gold prices for today",
            effort="minimal",
        ),
        [],
    )

    assert response.reply == forwardable
    assert messages == []


def test_save_history_includes_report_dir(tmp_path):
    from run_agents import ChatSession, _save_history

    history_path = tmp_path / "chat.json"
    report_dir = tmp_path / "reports" / "session"
    memory_dir = tmp_path / "memory" / "default"
    session = ChatSession(
        message_history=[ModelRequest.user_text_prompt("hello")],
        session_title="session",
        history_path=history_path,
        report_dir=report_dir,
        memory_dir=memory_dir,
    )

    _save_history(session)

    payload = json.loads(history_path.read_text())
    assert payload["session_title"] == "session"
    assert payload["report_dir"] == str(report_dir)
    assert payload["memory_dir"] == str(memory_dir)
    assert payload["messages"]


def test_memory_apply_accepts_explicit_low_risk_memory(tmp_path):
    from agents.runtime.memory import (
        MemoryExtraction,
        MemoryFinding,
        apply_memory_extraction,
        entry_path,
        events_path,
    )

    result = apply_memory_extraction(
        tmp_path,
        MemoryExtraction(
            findings=[
                MemoryFinding(
                    category="preference",
                    text="User prefers concise engineering answers.",
                    explicit=True,
                    confidence=0.95,
                    sensitivity="low",
                    reason="User directly stated a durable preference.",
                )
            ]
        ),
    )

    assert result.accepted == 1
    content = entry_path(tmp_path).read_text(encoding="utf-8")
    assert "## Preferences" in content
    assert "- User prefers concise engineering answers." in content
    events = [
        json.loads(line) for line in events_path(tmp_path).read_text().splitlines()
    ]
    assert events[0]["action"] == "accepted"


def test_memory_apply_keeps_inferred_memory_pending(tmp_path):
    from agents.runtime.memory import (
        MemoryExtraction,
        MemoryFinding,
        apply_memory_extraction,
        entry_path,
        pending_path,
    )

    result = apply_memory_extraction(
        tmp_path,
        MemoryExtraction(
            findings=[
                MemoryFinding(
                    category="project",
                    text="User is working on a local agent project.",
                    explicit=False,
                    confidence=0.8,
                    sensitivity="low",
                    reason="Inferred from the current task.",
                )
            ]
        ),
    )

    assert result.pending == 1
    assert not entry_path(tmp_path).exists()
    pending = [
        json.loads(line) for line in pending_path(tmp_path).read_text().splitlines()
    ]
    assert pending[0]["action"] == "pending"
    assert pending[0]["text"] == "User is working on a local agent project."


def test_memory_apply_accepts_explicit_high_sensitivity_memory(tmp_path):
    from agents.runtime.memory import (
        MemoryExtraction,
        MemoryFinding,
        apply_memory_extraction,
        entry_path,
        events_path,
    )

    result = apply_memory_extraction(
        tmp_path,
        MemoryExtraction(
            findings=[
                MemoryFinding(
                    category="environment",
                    text="User's API key is sk-test.",
                    explicit=True,
                    confidence=0.99,
                    sensitivity="high",
                    reason="Looks durable but contains API key sk-test.",
                )
            ]
        ),
    )

    assert result.accepted == 1
    content = entry_path(tmp_path).read_text(encoding="utf-8")
    assert "User's API key is sk-test." in content
    events = [
        json.loads(line) for line in events_path(tmp_path).read_text().splitlines()
    ]
    assert events[0]["action"] == "accepted"
    assert events[0]["sensitivity"] == "high"
    assert events[0]["text"] == "User's API key is sk-test."
    assert "sk-test" in events_path(tmp_path).read_text(encoding="utf-8")


def test_memory_apply_serializes_concurrent_updates(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from agents.runtime import memory

    first_rendering = threading.Event()
    second_rendered = threading.Event()
    original_render_entry = memory._render_entry

    def delayed_render_entry(prefix, sections, suffix):
        items = {item for section_items in sections.values() for item in section_items}
        if "User uses uv run." in items and "User uses zsh." not in items:
            first_rendering.set()
            second_rendered.wait(timeout=0.25)
        if "User uses zsh." in items:
            second_rendered.set()
        return original_render_entry(prefix, sections, suffix)

    def apply_finding(text):
        return memory.apply_memory_extraction(
            tmp_path,
            memory.MemoryExtraction(
                findings=[
                    memory.MemoryFinding(
                        category="environment",
                        text=text,
                        explicit=True,
                        confidence=0.95,
                        sensitivity="low",
                    )
                ]
            ),
        )

    monkeypatch.setattr(memory, "_render_entry", delayed_render_entry)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(apply_finding, "User uses uv run.")
        assert first_rendering.wait(timeout=1)
        second = executor.submit(apply_finding, "User uses zsh.")
        assert first.result(timeout=2).accepted == 1
        assert second.result(timeout=2).accepted == 1

    content = memory.entry_path(tmp_path).read_text(encoding="utf-8")
    assert "User uses uv run." in content
    assert "User uses zsh." in content


def test_memory_context_loads_entry_with_policy(tmp_path):
    from agents.runtime.memory import (
        MemoryExtraction,
        MemoryFinding,
        apply_memory_extraction,
        load_user_memory_context,
    )

    apply_memory_extraction(
        tmp_path,
        MemoryExtraction(
            findings=[
                MemoryFinding(
                    category="environment",
                    text="User uses zsh.",
                    explicit=True,
                    confidence=0.9,
                    sensitivity="low",
                )
            ]
        ),
    )

    context = load_user_memory_context(tmp_path)
    assert "Long-term user profile memory loaded from entry.md" in context
    assert "User uses zsh." in context
    assert "current user message overrides memory" in context.lower()


def test_memory_apply_findings_accepts_orchestrator_candidates(tmp_path):
    from agents.runtime import memory

    result = memory.apply_memory_findings(
        tmp_path,
        [
            memory.MemoryFinding(
                category="preference",
                text="User prefers concise answers.",
                explicit=True,
                confidence=0.95,
                sensitivity="low",
                reason="Orchestrator found a durable preference.",
            )
        ],
    )

    assert result.accepted == 1
    assert "User prefers concise answers." in memory.entry_path(tmp_path).read_text(
        encoding="utf-8"
    )


def test_memory_apply_findings_skips_empty_list(tmp_path):
    from agents.runtime import memory

    result = memory.apply_memory_findings(tmp_path, [])

    assert result.accepted == 0
    assert result.pending == 0
    assert result.rejected == 0
    assert not memory.entry_path(tmp_path).exists()
    assert not memory.events_path(tmp_path).exists()


def test_new_session_sets_report_dir(monkeypatch, tmp_path):
    import run_agents
    from run_agents import ChatSession, _init_session_paths_from_user_text

    monkeypatch.setattr(run_agents, "CHAT_HISTORY_DIR", tmp_path / "chats")
    monkeypatch.setattr(run_agents, "REPORT_ROOT", tmp_path / "reports")

    session = ChatSession()
    _init_session_paths_from_user_text(session, "read notes")

    assert session.session_title == "read-notes"
    assert session.report_dir == tmp_path / "reports" / "read-notes"


@pytest.mark.asyncio
async def test_run_turn_clears_report_dir_each_turn(monkeypatch, tmp_path):
    import run_agents
    from agents.orchestrator_agent import OrchestratorResponse
    from run_agents import ChatSession, run_turn

    report_dir = tmp_path / "reports" / "session"
    stale_report = report_dir / "fs-report.md"
    stale_report.parent.mkdir(parents=True)
    stale_report.write_text("stale report from previous turn", encoding="utf-8")
    captured: dict[str, str] = {}

    async def fake_run_orchestrator_turn(prompt, **_kwargs):
        captured["prompt"] = prompt
        assert report_dir.exists()
        assert not stale_report.exists()
        return SimpleNamespace(
            output=OrchestratorResponse(reply="ok"),
            decision=SimpleNamespace(memory_findings=[]),
            delegated=False,
            all_messages=lambda: [ModelRequest.user_text_prompt("hello")],
        )

    monkeypatch.setattr(run_agents, "run_orchestrator_turn", fake_run_orchestrator_turn)
    monkeypatch.setattr(run_agents, "scan_skills_context", lambda: "")
    monkeypatch.setattr(run_agents, "load_user_memory_context", lambda _memory_dir: "")

    session = ChatSession(
        session_title="session",
        history_path=tmp_path / "session.json",
        report_dir=report_dir,
        memory_dir=tmp_path / "memory",
    )

    await run_turn("hello", session)

    assert "stale report from previous turn" not in captured["prompt"]


@pytest.mark.asyncio
async def test_run_turn_loads_memory_once_as_orchestrator_context(
    monkeypatch, tmp_path
):
    import run_agents
    from agents.orchestrator_agent import OrchestratorResponse
    from agents.runtime.memory import (
        MemoryExtraction,
        MemoryFinding,
        apply_memory_extraction,
    )
    from run_agents import ChatSession, run_turn

    memory_dir = tmp_path / "memory"
    apply_memory_extraction(
        memory_dir,
        MemoryExtraction(
            findings=[
                MemoryFinding(
                    category="environment",
                    text="User keeps a private deployment key in memory.",
                    explicit=True,
                    confidence=0.95,
                    sensitivity="high",
                )
            ]
        ),
    )
    calls: list[tuple[str, str]] = []

    async def fake_run_orchestrator_turn(prompt, **kwargs):
        calls.append((prompt, kwargs.get("memory_context") or ""))
        return SimpleNamespace(
            output=OrchestratorResponse(reply="ok"),
            decision=SimpleNamespace(memory_findings=[]),
            delegated=False,
            all_messages=lambda: [ModelRequest.user_text_prompt(prompt)],
        )

    monkeypatch.setattr(run_agents, "run_orchestrator_turn", fake_run_orchestrator_turn)
    monkeypatch.setattr(run_agents, "scan_skills_context", lambda: "")

    session = ChatSession(
        session_title="session",
        history_path=tmp_path / "session.json",
        report_dir=tmp_path / "reports",
        memory_dir=memory_dir,
    )

    await run_turn("first", session)
    await run_turn("second", session)

    assert "private deployment key" not in calls[0][0]
    assert "private deployment key" in calls[0][1]
    assert "private deployment key" not in calls[1][0]
    assert calls[1][1] == ""


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


def test_fs_usage_limit_error_is_not_reported_as_file_access_problem():
    from pydantic_ai.exceptions import UsageLimitExceeded

    from agents.fs_agent import _format_exception_report
    from agents.runtime.reports import set_report_dir

    set_report_dir(None)
    message = _format_exception_report(
        "read local files",
        UsageLimitExceeded("The next tool call(s) would exceed the tool_calls_limit."),
    )

    assert "tool-call budget" in message
    assert "file access problem" not in message
