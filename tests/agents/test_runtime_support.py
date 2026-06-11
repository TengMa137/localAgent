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


@pytest.mark.asyncio
async def test_filesystem_duplicate_read_guard_rejects_same_call():
    from pydantic_ai.exceptions import ModelRetry

    from agents.runtime.context import DuplicateFilesystemReadGuardToolset

    calls: list[tuple[str, dict]] = []

    class Wrapped:
        async def call_tool(self, name, tool_args, _ctx, _tool):
            calls.append((name, tool_args))
            return "ok"

    guard = DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-1", messages=[])
    list_tool = SimpleNamespace(tool_def=SimpleNamespace(name="list_directory"))

    assert (
        await guard.call_tool(
            "list_directory",
            {"path": "/skills"},
            ctx,
            list_tool,
        )
        == "ok"
    )
    with pytest.raises(ModelRetry):
        await guard.call_tool(
            "list_directory",
            {"path": "/skills"},
            ctx,
            list_tool,
        )
    assert calls == [("list_directory", {"path": "/skills"})]


@pytest.mark.asyncio
async def test_filesystem_duplicate_read_guard_clears_after_write():
    from agents.runtime.context import DuplicateFilesystemReadGuardToolset

    class Wrapped:
        async def call_tool(self, _name, _tool_args, _ctx, _tool):
            return "ok"

    guard = DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-1", messages=[])
    list_tool = SimpleNamespace(tool_def=SimpleNamespace(name="list_directory"))
    write_tool = SimpleNamespace(tool_def=SimpleNamespace(name="write_file"))

    await guard.call_tool("list_directory", {"path": "/skills"}, ctx, list_tool)
    await guard.call_tool(
        "write_file",
        {"path": "/skills/a.md", "content": "x"},
        ctx,
        write_tool,
    )

    assert (
        await guard.call_tool(
            "list_directory",
            {"path": "/skills"},
            ctx,
            list_tool,
        )
        == "ok"
    )


@pytest.mark.asyncio
async def test_filesystem_guard_converts_read_discovery_error_to_model_retry():
    from pydantic_ai.exceptions import ModelRetry

    from agents.runtime.context import DuplicateFilesystemReadGuardToolset

    class Wrapped:
        async def call_tool(self, _name, _tool_args, _ctx, _tool):
            raise ValueError("outside validator boundaries")

    guard = DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-1", messages=[])
    list_tool = SimpleNamespace(tool_def=SimpleNamespace(name="list_directory"))

    with pytest.raises(ModelRetry, match="stop tool use"):
        await guard.call_tool("list_directory", {"path": ".skills"}, ctx, list_tool)


@pytest.mark.asyncio
async def test_filesystem_guard_stops_repeated_empty_discovery():
    from pydantic_ai.exceptions import ModelRetry

    from agents.runtime.context import (
        DuplicateFilesystemReadGuardToolset,
        MAX_EMPTY_DISCOVERY_CALLS,
    )

    class Wrapped:
        async def call_tool(self, _name, _tool_args, _ctx, _tool):
            return SimpleNamespace(count=0)

    guard = DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-1", messages=[])
    grep_tool = SimpleNamespace(tool_def=SimpleNamespace(name="grep_files"))

    for idx in range(MAX_EMPTY_DISCOVERY_CALLS - 1):
        assert (
            await guard.call_tool(
                "grep_files",
                {"path": "/docs", "query": f"missing-{idx}"},
                ctx,
                grep_tool,
            )
        ).count == 0

    with pytest.raises(ModelRetry, match="Multiple discovery searches"):
        await guard.call_tool(
            "grep_files",
            {"path": "/docs", "query": "missing-final"},
            ctx,
            grep_tool,
        )


@pytest.mark.asyncio
async def test_filesystem_task_scope_rejects_unrelated_mount_reads():
    from pydantic_ai.exceptions import ModelRetry

    from agents.runtime.context import (
        DuplicateFilesystemReadGuardToolset,
        filesystem_run_scope,
    )

    class Wrapped:
        async def call_tool(self, _name, _tool_args, _ctx, _tool):
            raise AssertionError("out-of-scope call must not reach filesystem")

    guard = DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-scoped-read", messages=[])
    read_tool = SimpleNamespace(tool_def=SimpleNamespace(name="read_file"))

    with filesystem_run_scope(["/docs"]):
        with pytest.raises(ModelRetry, match="outside this task's read scope"):
            await guard.call_tool(
                "read_file",
                {"path": "/skills/research/strategy.md"},
                ctx,
                read_tool,
            )


@pytest.mark.asyncio
async def test_filesystem_discovery_scope_rejects_full_candidate_reads():
    from pydantic_ai.exceptions import ModelRetry

    from agents.runtime.context import (
        DuplicateFilesystemReadGuardToolset,
        filesystem_run_scope,
    )

    class Wrapped:
        async def call_tool(self, _name, _tool_args, _ctx, _tool):
            raise AssertionError("full candidate read must not reach filesystem")

    guard = DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-preview-only", messages=[])
    read_tool = SimpleNamespace(tool_def=SimpleNamespace(name="read_file"))

    with filesystem_run_scope(["/docs"], discovery_preview_only=True):
        with pytest.raises(ModelRetry, match="preview_file"):
            await guard.call_tool(
                "read_file",
                {"path": "/docs/papers/world-model.md"},
                ctx,
                read_tool,
            )


@pytest.mark.asyncio
async def test_filesystem_topic_discovery_exposes_only_grep_and_preview():
    from agents.runtime.context import (
        DuplicateFilesystemReadGuardToolset,
        filesystem_run_scope,
    )

    class Wrapped:
        async def get_tools(self, _ctx):
            return {
                name: SimpleNamespace(tool_def=SimpleNamespace(name=name))
                for name in (
                    "grep_files",
                    "preview_file",
                    "find_paths",
                    "list_directory",
                    "read_file",
                )
            }

        async def call_tool(self, _name, tool_args, _ctx, _tool):
            return dict(tool_args)

    guard = DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-topic-tools", messages=[])
    grep_tool = SimpleNamespace(tool_def=SimpleNamespace(name="grep_files"))

    with filesystem_run_scope(
        ["/docs"],
        discovery_preview_only=True,
        discovery_search_paths=["/docs/papers/arxiv"],
    ) as run_state:
        initial_tools = await guard.get_tools(ctx)
        result = await guard.call_tool(
            "grep_files",
            {"query": "world model", "max_matches": 100},
            ctx,
            grep_tool,
        )
        next_tools = await guard.get_tools(ctx)

    assert set(initial_tools) == {"grep_files"}
    assert set(next_tools) == {"preview_file"}
    assert run_state.successful_calls == [
        (
            "grep_files",
            {
                "query": "world model",
                "max_matches": 12,
                "path": "/docs/papers/arxiv",
                "case_sensitive": False,
            },
        )
    ]
    assert result["path"] == "/docs/papers/arxiv"
    assert result["case_sensitive"] is False
    assert result["max_matches"] == 12


@pytest.mark.asyncio
async def test_filesystem_task_scope_rejects_copy_from_unrelated_mount(
    monkeypatch,
    tmp_path,
):
    from pydantic_ai.exceptions import ModelRetry

    from agents.runtime import context

    skills = tmp_path / "skills"
    skills.mkdir()
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[
                Mount(
                    host_path=skills,
                    mount_point="/skills",
                    mode="rw",
                )
            ]
        )
    )
    monkeypatch.setattr(context, "validator", validator)

    class Wrapped:
        async def call_tool(self, _name, _tool_args, _ctx, _tool):
            raise AssertionError("out-of-scope copy must not reach filesystem")

    guard = context.DuplicateFilesystemReadGuardToolset(Wrapped())
    ctx = SimpleNamespace(run_id="run-scoped-copy", messages=[])
    copy_tool = SimpleNamespace(tool_def=SimpleNamespace(name="copy_file"))

    with context.filesystem_run_scope(["/skills"]):
        with pytest.raises(ModelRetry, match="outside this task's read scope"):
            await guard.call_tool(
                "copy_file",
                {
                    "source": "/docs/secret.md",
                    "destination": "/skills/secret.md",
                },
                ctx,
                copy_tool,
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


def test_fs_preflight_matches_bare_filename_case_and_separator_insensitively(
    tmp_path,
):
    from agents.fs.path_policy import PathPreflight

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "AGENT_SYSTEM.md").write_text("Agent system notes.", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )

    sanitized, analysis = PathPreflight(
        ["/docs/AGENT_SYSTEM.md"],
        validator=validator,
    ).analyze("check agentsystem.md")

    assert sanitized == "check agentsystem.md"
    assert analysis.resolved_paths == ["/docs/AGENT_SYSTEM.md"]
    assert analysis.invalid_paths == []


def test_fs_system_prompt_keeps_only_model_owned_instructions():
    from agents.fs.prompts import FS_SYSTEM_PROMPT

    assert len(FS_SYSTEM_PROMPT) < 1800
    assert "topic_discovery" in FS_SYSTEM_PROMPT
    assert "grep_files first, then preview_file" in FS_SYSTEM_PROMPT
    assert "Do not call find_paths" in FS_SYSTEM_PROMPT
    assert "Return only the practical user-facing answer" in FS_SYSTEM_PROMPT
    assert "Python preflight" not in FS_SYSTEM_PROMPT
    assert "validator" not in FS_SYSTEM_PROMPT
    assert "Do not read the same file path twice" not in FS_SYSTEM_PROMPT


def test_fs_task_prompt_includes_path_facts_and_write_targets(monkeypatch):
    from agents import fs_agent

    monkeypatch.setattr(
        fs_agent,
        "_readable_file_index",
        lambda: [
            "/skills/fitness/diet.md",
            "/skills/fitness/workout.md",
            "/skills/skill_editing.md",
        ],
    )

    prompt, analysis = fs_agent._fs_task_prompt(
        "summarize the /skills/local-fitness-skills file"
    )
    assert analysis.invalid_paths == ["/skills/local-fitness-skills"]
    assert analysis.write_targets == []
    assert "Invalid path hints" in prompt
    assert "Missing-path chain:" in prompt
    assert "find_paths on the task scope (/skills)" in prompt
    assert "Readable file index" not in prompt

    prompt, analysis = fs_agent._fs_task_prompt(
        "create /skills/fitness/movement_recovery.md."
    )
    assert analysis.invalid_paths == ["/skills/fitness/movement_recovery.md"]
    assert analysis.write_targets == ["/skills/fitness/movement_recovery.md"]
    assert "New write targets" in prompt
    assert "- /skills/fitness/movement_recovery.md" in prompt
    assert "Mode: create_write" in prompt
    assert "Skill editing policy hook" in prompt
    assert "Source: /skills/skill_editing.md" in prompt


def test_fs_task_prompt_excludes_skills_for_ordinary_docs(monkeypatch):
    from agents import fs_agent

    monkeypatch.setattr(
        fs_agent,
        "_readable_file_index",
        lambda: [
            "/docs/papers/arxiv/world-model.md",
            "/skills/research/strategy.md",
        ],
    )
    monkeypatch.setattr(
        fs_agent,
        "scan_skills_context",
        lambda: (_ for _ in ()).throw(
            AssertionError("ordinary docs task must not scan skills")
        ),
    )

    prompt, analysis = fs_agent._fs_task_prompt(
        "check the local papers related to world models"
    )

    assert analysis.all_path_hints() == []
    assert "Scope: /docs" in prompt
    assert "/skills" not in prompt
    assert "strategy.md" not in prompt
    task_roots = fs_agent._task_read_roots(
        "check the local papers related to world models",
        analysis,
    )
    assert fs_agent._preemptive_rag_paths(
        "check the local papers related to world models",
        analysis,
        task_roots,
    ) == []
    assert "Mode: topic_discovery" in prompt
    assert "Search path: /docs" in prompt


def test_fs_task_prompt_requires_model_tool_discovery_for_ambiguous_artifact(
    monkeypatch,
):
    from agents import fs_agent

    monkeypatch.setattr(
        fs_agent,
        "_readable_file_index",
        lambda: ["/docs/papers/agent-architecture.md"],
    )

    prompt, analysis = fs_agent._fs_task_prompt(
        "summarize the paper and explain its architecture"
    )

    assert analysis.all_path_hints() == []
    assert "Mode: topic_discovery" in prompt
    assert "Search path: /docs" in prompt
    assert "web recovery is allowed" in prompt


def test_fs_task_prompt_allows_web_recovery_for_local_paper_lookup(monkeypatch):
    from agents import fs_agent

    monkeypatch.setattr(
        fs_agent,
        "_readable_file_index",
        lambda: ["/docs/papers/world-model.md"],
    )

    prompt, _analysis = fs_agent._fs_task_prompt(
        "check local papers regarding recent world model research"
    )

    assert "Mode: topic_discovery" in prompt
    assert "Search path: /docs" in prompt
    assert "web recovery is allowed" in prompt


def test_fs_topic_lookup_inside_explicit_directory_still_uses_lexical_triage(
    monkeypatch,
    tmp_path,
):
    from agents import fs_agent

    docs = tmp_path / "docs"
    arxiv = docs / "papers" / "arxiv"
    arxiv.mkdir(parents=True)
    (arxiv / "world-model.md").write_text(
        "# World Model\n\n## Abstract\nPredictive dynamics.",
        encoding="utf-8",
    )
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    objective = "check papers related to world models under /docs/arxiv"
    prompt, analysis = fs_agent._fs_task_prompt(objective)
    task_roots = fs_agent._task_read_roots(objective, analysis)

    assert analysis.resolved_paths == ["/docs/papers/arxiv"]
    assert analysis.invalid_paths == []
    assert "Mode: topic_discovery" in prompt
    assert "Search path: /docs/papers/arxiv" in prompt
    assert fs_agent._requires_lexical_triage(objective, analysis) is True
    assert fs_agent._preemptive_rag_paths(
        objective,
        analysis,
        task_roots,
    ) == []


def test_fs_non_paper_prompt_keeps_all_docs_in_scope(monkeypatch, tmp_path):
    from agents import fs_agent

    docs = tmp_path / "docs"
    papers = docs / "papers"
    papers.mkdir(parents=True)
    (docs / "agentsystem.md").write_text("Agent system notes.", encoding="utf-8")
    (docs / "auth.md").write_text("Authentication notes.", encoding="utf-8")
    (papers / "world-model.md").write_text("World model paper.", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    prompt, analysis = fs_agent._fs_task_prompt(
        "find the local authentication documentation and summarize it"
    )
    task_roots = fs_agent._task_read_roots(
        "find the local authentication documentation and summarize it",
        analysis,
    )

    assert task_roots == ["/docs"]
    assert "Mode: topic_discovery" in prompt
    assert "Search path: /docs" in prompt
    assert "Readable file index" not in prompt
    assert fs_agent._preemptive_rag_paths(
        "find the local authentication documentation and summarize it",
        analysis,
        task_roots,
    ) == []


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


def test_fs_success_response_keeps_full_findings_out_of_tool_return():
    from agents.fs_agent import FsAgentResult, _format_success_response

    result = FsAgentResult(
        answer="Use this answer directly.",
        summary="Short filesystem summary.",
        paths=["/docs/a.md"],
        findings=["long finding that should stay out of the compact handoff"],
        changes_made=["updated /docs/a.md"],
    )

    formatted = _format_success_response(result)

    assert "Forwardable answer:" in formatted
    assert "Use this answer directly." in formatted
    assert "Summary: Short filesystem summary." in formatted
    assert "Detailed findings: 1 item(s)" in formatted
    assert "long finding that should stay out of the compact handoff" not in formatted
    assert "updated /docs/a.md" not in formatted


def test_web_response_keeps_full_findings_out_of_tool_return():
    from agents.web_agent import WebAgentResult, _format_orchestrator_response

    result = WebAgentResult(
        answer="Current answer for the user.",
        summary="Short web summary.",
        search_queries=["example query"],
        urls=["https://example.com"],
        findings=["large crawled/RAG finding that should stay out of the compact handoff"],
    )

    formatted = _format_orchestrator_response(result)

    assert "Forwardable answer:" in formatted
    assert "Current answer for the user." in formatted
    assert "Summary: Short web summary." in formatted
    assert "Detailed findings: 1 item(s)" in formatted
    assert "large crawled/RAG finding that should stay out of the compact handoff" not in formatted


def test_web_query_guidance_includes_time_sensitive_semantic_guidance():
    from agents.web_agent import _web_query_guidance

    guidance = _web_query_guidance("What's today's gold price?")

    assert "Current date/time:" in guidance
    assert (
        "Choose the web search query semantically from the objective"
        in guidance
    )
    assert "avoid adding a bare year" in guidance


def test_scholarly_result_urls_prefers_newest_arxiv_id():
    from agents.web.policy import scholarly_result_urls

    results = [
        {
            "url": "https://arxiv.org/abs/2504.13152",
            "position": 1,
        },
        {
            "url": "https://arxiv.org/abs/2602.10094",
            "position": 3,
        },
    ]

    assert scholarly_result_urls(results, prefer_recent=True) == [
        "https://arxiv.org/html/2602.10094",
        "https://arxiv.org/html/2504.13152",
    ]


def test_save_arxiv_markdown_documents_writes_under_docs(tmp_path):
    from agents.web.paper_storage import save_arxiv_markdown_documents
    from rag import Document

    paths = save_arxiv_markdown_documents(
        [
            Document(
                doc_id="paper-1",
                source="https://arxiv.org/html/2602.10094v2",
                title=(
                    "arxiv.org — 4RC: 4D Reconstruction via Conditional Querying"
                ),
                text=(
                    "# 4RC: 4D Reconstruction via Conditional Querying\n\n"
                    "Full crawled paper content."
                ),
                mime="text/markdown",
                meta={},
            )
        ],
        docs_dir=tmp_path,
    )

    saved = tmp_path / "papers" / "arxiv" / "2602.10094v2.md"
    assert paths == ["/docs/papers/arxiv/2602.10094v2.md"]
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "# 4RC: 4D Reconstruction via Conditional Querying" in content
    assert "- arXiv ID: 2602.10094v2" in content
    assert "Full crawled paper content." in content


def test_orchestrator_prompt_explicitly_routes_current_lookup_to_web():
    from agents.orchestrator_agent import _orchestrator_prompt_body

    prompt = _orchestrator_prompt_body()

    assert "If the user explicitly asks you to search" in prompt
    assert "Live market prices" in prompt
    assert "changing facts are web tasks" in prompt
    assert "paper discovery" in prompt
    assert "assistant message explicitly saved under" in prompt
    assert "/docs path" in prompt
    assert "fetch/download/save the paper locally" in prompt
    assert "source-ownership test" in prompt
    assert "one dedicated external API lookup" in prompt
    assert "web specialist chooses its" in prompt
    assert "Do not choose plan merely because one web lookup" in prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "cool, now fetch me recent language model research",
        (
            "check recent paper about 4d reconstruction online and fetch one "
            "you think is most valuable"
        ),
        "find the recent paper online regarding 4d reconstruction",
    ],
)
def test_orchestrator_guardrail_corrects_recent_research_fs_to_web(prompt):
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
    )

    decision = OrchestratorDecision(
        route="fs",
        objective=(
            "Provide recent and comprehensive overview of language model research "
            "including key trends and breakthroughs"
        ),
    )

    corrected = _guardrail_orchestrator_decision(
        prompt,
        decision,
    )

    assert corrected.route == "web"
    assert corrected.objective == prompt


def test_orchestrator_guardrail_keeps_local_docs_paper_followup_on_fs():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
    )

    decision = OrchestratorDecision(
        route="fs",
        objective="Summarize /docs/papers/arxiv/2605.06548v1.md experiments",
    )

    corrected = _guardrail_orchestrator_decision(
        "break down the paper at /docs/papers/arxiv/2605.06548v1.md",
        decision,
    )

    assert corrected.route == "fs"
    assert corrected.objective == decision.objective


def test_orchestrator_guardrail_keeps_relative_docs_path_on_fs():
    from agents.orchestrator_agent import (
        OrchestratorChoice,
        _decision_from_choice,
        _guardrail_orchestrator_decision,
        _mentioned_validator_roots,
    )

    prompt = "summarize agentsystem.md under docs/"
    decision = _decision_from_choice(
        prompt,
        OrchestratorChoice(
            route="fs",
            content="Summarize the current agent system documentation.",
        ),
    )
    corrected = _guardrail_orchestrator_decision(prompt, decision)

    assert _mentioned_validator_roots(prompt) == {"/docs"}
    assert decision.objective == prompt
    assert corrected.route == "fs"
    assert corrected.objective == prompt


def test_orchestrator_guardrail_prefers_explicit_filename_over_discourse_now():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
        _orchestrator_model_prompt,
    )

    prompt = (
        "ok, now check agentsystem.md and tell me how to build a robust "
        "agent system"
    )
    corrected = _guardrail_orchestrator_decision(
        prompt,
        OrchestratorDecision(
            route="web",
            objective="Search for current agent system guidance",
        ),
    )

    assert corrected.route == "fs"
    assert corrected.objective == prompt
    model_prompt = _orchestrator_model_prompt(prompt)
    assert "explicit local filename(s): agentsystem.md" in model_prompt
    assert len(model_prompt) < len(prompt) + 150


def test_orchestrator_guardrail_does_not_turn_discourse_now_into_web():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
    )

    decision = OrchestratorDecision(
        route="direct",
        reply="A robust agent system uses narrow contracts and deterministic guards.",
    )

    assert (
        _guardrail_orchestrator_decision(
            "ok, now explain how to build a robust agent system",
            decision,
        )
        == decision
    )


def test_orchestrator_guardrail_routes_ambiguous_artifact_local_first():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
        _orchestrator_model_prompt,
    )

    prompt = "summarize the paper and explain its architecture"
    corrected = _guardrail_orchestrator_decision(
        prompt,
        OrchestratorDecision(
            route="web",
            objective="Search online for the paper",
        ),
    )

    assert corrected.route == "fs"
    assert corrected.objective == prompt
    assert "Try fs discovery first" in _orchestrator_model_prompt(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "now summarize all papers locally in parallel",
        "I mean all papers under docs/papers/arxiv",
        (
            "you should first check all papers locally then read and summarize "
            "them in parallel"
        ),
    ],
)
def test_orchestrator_guardrail_routes_collection_work_to_plan(prompt):
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
        _orchestrator_model_prompt,
    )

    corrected = _guardrail_orchestrator_decision(
        prompt,
        OrchestratorDecision(route="direct", reply="One paper summary."),
    )

    assert corrected.route == "plan"
    assert corrected.objective == prompt
    assert corrected.effort == "standard"
    assert "collection-wide work" in _orchestrator_model_prompt(prompt)


def test_orchestrator_guardrail_preserves_exact_collection_request():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
    )

    prompt = "I mean all papers under docs/papers/arxiv"
    corrected = _guardrail_orchestrator_decision(
        prompt,
        OrchestratorDecision(
            route="plan",
            objective=(
                "Plan to read all papers under docs/papers/arxiv and "
                "process/batch the files."
            ),
            effort="standard",
        ),
    )

    assert corrected.route == "plan"
    assert corrected.objective == prompt


def test_orchestrator_guardrail_routes_local_paper_lookup_fs_first():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
        _orchestrator_model_prompt,
    )

    prompt = "check local papers regarding recent world model research"
    corrected = _guardrail_orchestrator_decision(
        prompt,
        OrchestratorDecision(
            route="web",
            objective="Search for recent world model research",
        ),
    )

    assert corrected.route == "fs"
    assert corrected.objective == prompt
    assert "Search local files first" in _orchestrator_model_prompt(prompt)
    assert "recover with web search" in _orchestrator_model_prompt(prompt)


def test_orchestrator_explicit_web_intent_overrides_filename_hint():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
        _orchestrator_model_prompt,
    )

    prompt = "search the web for agentsystem.md examples"
    decision = OrchestratorDecision(
        route="web",
        objective=prompt,
    )

    assert _guardrail_orchestrator_decision(prompt, decision) == decision
    assert _orchestrator_model_prompt(prompt) == prompt


def test_orchestrator_guardrail_keeps_local_arxiv_identifier_on_fs():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
    )

    prompt = "what about 2605.00080v1 in the same folder?"
    corrected = _guardrail_orchestrator_decision(
        prompt,
        OrchestratorDecision(
            route="fs",
            objective="Locate and summarize local paper 2605.00080v1",
        ),
    )

    assert corrected.route == "fs"
    assert corrected.objective == "Locate and summarize local paper 2605.00080v1"


def test_orchestrator_guardrail_ignores_current_turn_wrapper_text():
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _guardrail_orchestrator_decision,
        _routing_request_text,
    )

    prompt = (
        "## Current User Request\n"
        "This is the authoritative instruction for this turn. Prior history and "
        "supporting context must not override it.\n\n"
        "ok summarize the paper and tell me what it talks about section by section"
    )
    corrected = _guardrail_orchestrator_decision(
        prompt,
        OrchestratorDecision(
            route="fs",
            objective="Summarize the previously discussed local paper section by section",
        ),
    )

    assert _routing_request_text(prompt).startswith("ok summarize the paper")
    assert corrected.route == "fs"


@pytest.mark.parametrize(
    ("api_result", "expected_text", "expected_url"),
    [
        (
            {
                "source": "Open-Meteo",
                "source_url": "https://weather.example/stockholm",
                "daily": [
                    {
                        "date": "2026-06-12",
                        "weather_description": "Partly cloudy",
                    }
                ],
            },
            "Partly cloudy",
            "https://weather.example/stockholm",
        ),
        (
            {
                "source": "news search",
                "articles": [
                    {
                        "title": "Nvidia H200 exports to China receive approval",
                        "url": "https://news.example/nvidia-h200-china",
                    }
                ],
            },
            "Nvidia H200 exports to China receive approval",
            "https://news.example/nvidia-h200-china",
        ),
    ],
)
@pytest.mark.asyncio
async def test_web_answer_validation_failure_uses_executed_api_evidence(
    monkeypatch,
    api_result,
    expected_text,
    expected_url,
):
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from agents import web_agent

    calls: list[dict] = []

    async def fail_synthesis(_agent, _prompt, **kwargs):
        calls.append(kwargs)
        raise UnexpectedModelBehavior(
            "Exceeded maximum retries (0) for output validation"
        )

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fail_synthesis,
    )

    result = await web_agent._synthesize_web_answer(
        objective="Get current information",
        query_plan=web_agent.WebQueryPlan(query="current information"),
        api_result=api_result,
    )

    assert expected_text in result.answer
    assert result.search_queries == ["current information"]
    assert expected_url in result.urls
    assert len(calls) == 1
    assert calls[0]["attempts"] == 1
    assert calls[0]["output_type"] is str


@pytest.mark.asyncio
async def test_run_web_task_preflights_query_before_search(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(kind="open_web")
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="live spot gold price today USD per ounce",
                    as_of="Thursday, 04 June 2026, 19:00 UTC",
                    search_result_limit=3,
                    crawl_url_limit=1,
                    checks=["Matches current live gold spot price objective."],
                )
            )
        if output_name == "WebPreviewDecision":
            assert "as_of: Saturday, 06 June 2026, 15:00 UTC" in _prompt
            assert "Thursday, 04 June 2026, 19:00 UTC" not in _prompt
            return SimpleNamespace(
                output=web_agent.WebPreviewDecision(
                    answer_from_preview=False,
                    selected_urls=["https://example.com/gold"],
                    reason="Need one source page for verification.",
                )
            )
        return SimpleNamespace(output="Gold is quoted at the tested value.")

    async def fake_search(_mcp_url, query, *, max_results=None):
        events.append(("search", query))
        assert events[-2] == ("model", "WebQueryPlan")
        assert max_results == 3
        return [
            {
                "title": "Gold Price Today",
                "url": "https://example.com/gold",
                "snippet": "Live spot gold quote.",
                "position": 1,
            }
        ]

    async def fake_crawl(_mcp_url, _rag_service, urls):
        events.append(("crawl", ",".join(urls)))
        return "Ingested 1 document(s): example."

    async def fake_rag_search(**_kwargs):
        events.append(("rag", "search"))
        return [
            {
                "node_id": "node-1",
                "source": "https://example.com/gold",
                "title": "Gold Price Today",
                "text": "Live spot gold quote.",
            }
        ]

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(
        web_agent,
        "_now",
        lambda: "Saturday, 06 June 2026, 15:00 UTC",
    )
    monkeypatch.setattr(web_agent, "web_search_results", fake_search)
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fake_crawl)
    monkeypatch.setattr(web_agent, "rag_search_documents", fake_rag_search)

    result = await web_agent.run_web_task("check the gold price on today")

    assert "Gold is quoted at the tested value." in result
    assert events == [
        ("model", "WebSourceDecision"),
        ("model", "WebQueryPlan"),
        ("search", "live spot gold price today USD per ounce"),
        ("model", "WebPreviewDecision"),
        ("crawl", "https://example.com/gold"),
        ("rag", "search"),
        ("model", "web answer text"),
    ]


@pytest.mark.asyncio
async def test_run_web_task_skips_crawl_when_preview_is_enough(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(kind="open_web")
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="weather Lund Sweden tomorrow",
                    as_of="Thursday, 04 June 2026, 21:40 UTC",
                    search_result_limit=3,
                    crawl_url_limit=0,
                    checks=[
                        "Tomorrow means Friday, 05 June 2026 for Lund, Sweden."
                    ],
                )
            )
        if output_name == "WebPreviewDecision":
            return SimpleNamespace(
                output=web_agent.WebPreviewDecision(
                    answer_from_preview=True,
                    reason="Snippet includes the forecast for the requested city/date.",
                )
            )
        return SimpleNamespace(
            output="Lund, Sweden is forecast to have passing showers tomorrow."
        )

    async def fake_search(_mcp_url, query, *, max_results=None):
        events.append(("search", f"{query}:{max_results}"))
        assert max_results == 3
        return [
            {
                "title": "Lund weather forecast",
                "url": "https://weather.example/lund",
                "snippet": (
                    "Friday, June 5: passing showers with breaks of sun late."
                ),
                "position": 1,
            }
        ]

    async def fail_crawl(*_args, **_kwargs):
        raise AssertionError("crawl should be skipped when preview is enough")

    async def fail_rag(*_args, **_kwargs):
        raise AssertionError("RAG should be skipped when crawl is skipped")

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "web_search_results", fake_search)
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fail_crawl)
    monkeypatch.setattr(web_agent, "rag_search_documents", fail_rag)

    result = await web_agent.run_web_task(
        "then check the weather in Lund, sweden tomorrow"
    )

    assert "passing showers" in result
    assert events == [
        ("model", "WebSourceDecision"),
        ("model", "WebQueryPlan"),
        ("search", "weather Lund Sweden tomorrow:3"),
        ("model", "WebPreviewDecision"),
        ("model", "web answer text"),
    ]


@pytest.mark.asyncio
async def test_run_web_task_uses_source_domain_queries(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(kind="open_web")
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="weather Lund Sweden tomorrow",
                    as_of="Thursday, 04 June 2026, 21:40 UTC",
                    source_domains=["timeanddate.com/weather", "weather.com"],
                    search_result_limit=5,
                    crawl_url_limit=0,
                )
            )
        if output_name == "WebPreviewDecision":
            return SimpleNamespace(
                output=web_agent.WebPreviewDecision(
                    answer_from_preview=True,
                    reason="Weather previews are enough.",
                )
            )
        return SimpleNamespace(
            output="Lund weather answered from source-scoped previews."
        )

    async def fake_search(_mcp_url, query, *, max_results=None):
        events.append(("search", f"{query}:{max_results}"))
        assert max_results == 3
        return [
            {
                "title": "Lund weather",
                "url": f"https://example.com/{len(events)}",
                "snippet": "Forecast preview.",
                "position": len(events),
            }
        ]

    async def fail_crawl(*_args, **_kwargs):
        raise AssertionError("crawl should be skipped when preview is enough")

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "web_search_results", fake_search)
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fail_crawl)

    result = await web_agent.run_web_task("check Lund weather tomorrow")

    assert "Lund weather answered" in result
    assert (
        "search",
        "weather Lund Sweden tomorrow:3",
    ) in events
    assert (
        "search",
        "site:timeanddate.com/weather weather Lund Sweden tomorrow:3",
    ) in events
    assert (
        "search",
        "site:weather.com weather Lund Sweden tomorrow:3",
    ) in events
    assert "site:timeanddate.com/weather weather Lund Sweden tomorrow" in result


@pytest.mark.asyncio
async def test_run_web_task_uses_weather_forecast_mcp_tool(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(
                    kind="weather",
                    target="Lund, Sweden",
                )
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="Lund, Sweden - tomorrow's weather forecast (Saturday, 06 June 2026)",
                    preferred_source="weather_forecast",
                    date="2026-06-06",
                    search_result_limit=3,
                    crawl_url_limit=0,
                    checks=["Tomorrow resolves to 2026-06-06 in Lund."],
                )
            )
        return SimpleNamespace(output="Lund forecast: partly cloudy, 18 C high.")

    async def fake_weather(_mcp_url, location, *, date=None):
        events.append(("weather", f"{location}:{date}"))
        return {
            "success": True,
            "query": location,
            "date": date,
            "daily": {
                "date": date,
                "weather_description": "Partly cloudy",
                "temperature_max_c": 18,
            },
            "source_url": "https://open-meteo.com/en/docs",
        }

    async def fail_search(*_args, **_kwargs):
        raise AssertionError("weather API path should not call web search")

    async def fail_crawl(*_args, **_kwargs):
        raise AssertionError("weather API path should not call crawl")

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "weather_forecast_result", fake_weather)
    monkeypatch.setattr(web_agent, "web_search_results", fail_search)
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fail_crawl)

    result = await web_agent.run_web_task("check the weather in Lund tomorrow")

    assert "Lund forecast" in result
    assert ("weather", "Lund, Sweden:2026-06-06") in events
    assert ("model", "WebPreviewDecision") not in events


def test_preferred_api_tool_accepts_exact_tool_name_or_source_alias():
    from agents.web_agent import (
        WebQueryPlan,
        WebSourceDecision,
        _preferred_api_tool,
    )

    assert (
        _preferred_api_tool(
            WebQueryPlan(
                query="Lund weather tomorrow",
                preferred_source="weather_forecast",
            )
        )
        == "weather_forecast"
    )
    assert (
        _preferred_api_tool(
            WebQueryPlan(
                query="Lund weather tomorrow",
                preferred_source="weather",
            )
        )
        == "weather_forecast"
    )
    assert (
        _preferred_api_tool(
            WebQueryPlan(
                query="Lund weather tomorrow",
                preferred_tool="WEATHER_FORECAST",
            )
        )
        == "weather_forecast"
    )
    assert WebSourceDecision(kind="recent_news").method == "web"
    assert WebSourceDecision(kind="scholarly").method == "web"


@pytest.mark.asyncio
async def test_run_web_task_uses_wiki_api_without_search_or_crawl(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []
    objective = "look up an encyclopedia overview of the printing press"
    answer = "The printing press is a mechanical printing technology."
    api_payload = {
        "success": True,
        "title": "Printing press",
        "extract": "A printing press applies pressure to an inked surface.",
        "page_url": "https://en.wikipedia.org/wiki/Printing_press",
    }

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(
                    kind="encyclopedia",
                    target="Printing press",
                )
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query=objective,
                    preferred_source="wiki_summary",
                    preferred_tool=None,
                    search_result_limit=3,
                    crawl_url_limit=0,
                )
            )
        return SimpleNamespace(output=answer)

    async def fake_wiki(_mcp_url, query, *, language=None):
        events.append(("wiki", f"{query}:{language}"))
        return api_payload

    async def fail_search(*_args, **_kwargs):
        raise AssertionError("structured API path should not call web search")

    async def fail_crawl(*_args, **_kwargs):
        raise AssertionError("structured API path should not crawl")

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "wiki_summary_result", fake_wiki)
    monkeypatch.setattr(web_agent, "web_search_results", fail_search)
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fail_crawl)

    result = await web_agent.run_web_task(objective)

    assert answer in result
    assert ("model", "WebPreviewDecision") not in events
    assert ("wiki", "Printing press:None") in events


@pytest.mark.asyncio
async def test_empty_structured_api_result_falls_back_to_one_web_search(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(
                    kind="encyclopedia",
                    target="Example Agency",
                )
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="current office holder for example agency",
                    preferred_source="wiki_summary",
                    crawl_url_limit=0,
                )
            )
        if output_name == "WebPreviewDecision":
            return SimpleNamespace(
                output=web_agent.WebPreviewDecision(
                    answer_from_preview=True,
                    reason="Official-source snippet contains the current office holder.",
                )
            )
        return SimpleNamespace(
            output="The current office holder was found from web results."
        )

    async def fake_wiki(*_args, **_kwargs):
        events.append(("wiki", "empty"))
        return {"success": False, "error": "page not found"}

    async def fake_search(_mcp_url, query, *, max_results=None):
        events.append(("search", f"{query}:{max_results}"))
        return [
            {
                "title": "Example Agency leadership",
                "url": "https://example.gov/leadership",
                "snippet": "The current office holder is Example Person.",
                "position": 1,
            }
        ]

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "wiki_summary_result", fake_wiki)
    monkeypatch.setattr(web_agent, "web_search_results", fake_search)

    result = await web_agent.run_web_task("who currently leads the example agency?")

    assert "current office holder was found from web results" in result
    assert ("wiki", "empty") in events
    assert ("search", "current office holder for example agency:5") in events
    assert events.count(("model", "WebPreviewDecision")) == 1


@pytest.mark.asyncio
async def test_recent_news_uses_web_search_directly(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(
                    kind="recent_news",
                    target="EU AI Act",
                )
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="latest EU AI Act reporting",
                    search_result_limit=3,
                    crawl_url_limit=0,
                )
            )
        if output_name == "WebPreviewDecision":
            return SimpleNamespace(
                output=web_agent.WebPreviewDecision(answer_from_preview=True)
            )
        return SimpleNamespace(
            output="Fallback search found current EU AI Act reporting."
        )

    async def fake_search(_mcp_url, query, *, max_results=None):
        events.append(("search", f"{query}:{max_results}"))
        return [
            {
                "title": "EU AI Act update",
                "url": "https://example.com/eu-ai-act",
                "snippet": "A current update.",
                "position": 1,
            }
        ]

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "web_search_results", fake_search)

    result = await web_agent.run_web_task("find the latest EU AI Act reporting")

    assert "Fallback search found" in result
    assert ("search", "latest EU AI Act reporting:3") in events
    assert all(event[0] != "news" for event in events)


@pytest.mark.asyncio
async def test_run_web_task_normalizes_weather_location_without_model_retry(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(
                    kind="weather",
                )
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="Lund, Sweden - tomorrow's weather forecast (Saturday, 06 June 2026)",
                    preferred_tool="weather_forecast",
                    date="2026-06-06",
                    search_result_limit=3,
                    crawl_url_limit=0,
                )
            )
        return SimpleNamespace(output="Lund weather lookup succeeded.")

    async def fake_weather(_mcp_url, location, *, date=None):
        events.append(("weather", f"{location}:{date}"))
        if location != "Lund, Sweden":
            return {
                "success": False,
                "query": location,
                "date": date,
                "error": "location not found",
            }
        return {
            "success": True,
            "query": location,
            "date": date,
            "daily": {"weather_description": "Partly cloudy"},
        }

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "weather_forecast_result", fake_weather)

    result = await web_agent.run_web_task("check the weather in Lund tomorrow")

    assert "lookup succeeded" in result
    assert ("weather", "Lund, Sweden:2026-06-06") in events



@pytest.mark.asyncio
async def test_run_web_task_searches_arxiv_through_generic_web_flow(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(
                    kind="scholarly",
                    target="diffusion language models",
                )
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query="diffusion language models",
                    search_result_limit=4,
                    crawl_url_limit=0,
                )
            )
        if output_name == "WebPreviewDecision":
            return SimpleNamespace(
                output=web_agent.WebPreviewDecision(
                    answer_from_preview=True,
                    reason="The result preview identifies the requested paper.",
                )
            )
        return SimpleNamespace(output="Large Language Diffusion Models is relevant.")

    async def fake_web_search(_mcp_url, query, *, max_results=None):
        events.append(("web_search", query))
        assert max_results == 3
        return [
            {
                "title": "[2502.09992] Large Language Diffusion Models",
                "url": "https://arxiv.org/abs/2502.09992",
                "snippet": "Introduces LLaDA.",
                "position": 1,
            }
        ]

    async def fail_crawl(*_args, **_kwargs):
        raise AssertionError("preview-sufficient scholarly search must not crawl")

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "web_search_results", fake_web_search)
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fail_crawl)

    result = await web_agent.run_web_task(
        "search arXiv for diffusion language models"
    )

    assert "Large Language Diffusion Models is relevant." in result
    assert ("web_search", "site:arxiv.org diffusion language models") in events
    assert ("model", "ArxivSelectionDecision") not in events


@pytest.mark.asyncio
async def test_scholarly_fetch_grounds_query_and_crawls_returned_paper(monkeypatch):
    from agents import web_agent

    events: list[tuple[str, str]] = []
    hallucinated = (
        "Learning to reconstruct 4D volumetric scenes from single RGB-D images "
        "using a multi-view geometric prior (2023) Zhang et al."
    )

    async def fake_model_run(_agent, prompt, *, output_name, **_kwargs):
        events.append(("model", output_name))
        if output_name == "WebSourceDecision":
            return SimpleNamespace(
                output=web_agent.WebSourceDecision(
                    kind="scholarly",
                    target=hallucinated,
                )
            )
        if output_name == "WebQueryPlan":
            return SimpleNamespace(
                output=web_agent.WebQueryPlan(
                    query=hallucinated,
                    search_result_limit=5,
                    crawl_url_limit=0,
                )
            )
        if output_name == "WebPreviewDecision":
            return SimpleNamespace(
                output=web_agent.WebPreviewDecision(
                    answer_from_preview=True,
                    selected_urls=["https://arxiv.org/abs/2507.21045"],
                    reason="A snippet appears sufficient.",
                )
            )
        assert hallucinated not in prompt
        assert "4RC: 4D Reconstruction via Conditional Querying" in prompt
        assert "D4RT: Dynamic 4D Reconstruction and Tracking" not in prompt
        return SimpleNamespace(
            output="4RC is a recent paper on feed-forward 4D reconstruction."
        )

    async def fake_web_search(_mcp_url, query, *, max_results=None):
        events.append(("web_search", query))
        assert hallucinated not in query
        assert max_results == 3
        return [
            {
                "title": "4RC: 4D Reconstruction via Conditional Querying",
                "url": "https://arxiv.org/abs/2602.10094",
                "snippet": "A unified feed-forward framework for 4D reconstruction.",
                "position": 2,
            },
            {
                "title": "D4RT: Dynamic 4D Reconstruction and Tracking",
                "url": "https://arxiv.org/abs/2507.21045",
                "snippet": "A unified 4D reconstruction and tracking model.",
                "position": 1,
            }
        ]

    async def fake_crawl(
        _mcp_url,
        _rag_service,
        urls,
        *,
        capture_documents=None,
    ):
        events.append(("crawl", ",".join(urls)))
        if urls == ["https://arxiv.org/html/2602.10094"]:
            return "No usable content retrieved from the arXiv HTML page."
        assert urls == ["https://arxiv.org/abs/2602.10094"]
        assert capture_documents is not None
        capture_documents.append(
            SimpleNamespace(
                source="https://arxiv.org/abs/2602.10094",
                title="arxiv.org — 4RC",
                text="Full paper content.",
            )
        )
        return "Ingested 1 document(s): 4RC."

    async def fake_rag_search(**kwargs):
        assert kwargs["docs"] == ["https://arxiv.org/abs/2602.10094"]
        return [
            {
                "node_id": "4rc-abstract",
                "source": "https://arxiv.org/abs/2602.10094",
                "title": "4RC: 4D Reconstruction via Conditional Querying",
                "text": "4RC jointly captures dense scene geometry and motion dynamics.",
            }
        ]

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "web_search_results", fake_web_search)
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fake_crawl)
    monkeypatch.setattr(web_agent, "rag_search_documents", fake_rag_search)
    monkeypatch.setattr(
        web_agent,
        "_save_arxiv_documents",
        lambda documents: (
            ["/docs/papers/arxiv/2602.10094.md"] if documents else []
        ),
    )

    result = await web_agent.run_web_task(
        "search online the recent paper on 4d reconstruction and fetch one for me"
    )

    assert "4RC is a recent paper" in result
    assert ("web_search", "recent paper on 4d reconstruction") in events
    assert (
        "web_search",
        "site:arxiv.org recent paper on 4d reconstruction",
    ) in events
    assert ("crawl", "https://arxiv.org/html/2602.10094") in events
    assert ("crawl", "https://arxiv.org/abs/2602.10094") in events
    assert "Saved locally: `/docs/papers/arxiv/2602.10094.md`" in result


@pytest.mark.asyncio
async def test_direct_arxiv_fetch_falls_back_and_saves_locally(monkeypatch):
    from agents import web_agent

    crawls: list[str] = []

    async def fake_model_run(_agent, _prompt, *, output_name, **_kwargs):
        assert output_name == "web answer text"
        return SimpleNamespace(output="Fetched the requested arXiv paper.")

    async def fake_crawl(
        _mcp_url,
        _rag_service,
        urls,
        *,
        capture_documents=None,
    ):
        crawls.extend(urls)
        if urls == ["https://arxiv.org/html/2602.10094"]:
            return "No usable content retrieved from the arXiv HTML page."
        assert urls == ["https://arxiv.org/abs/2602.10094"]
        capture_documents.append(
            SimpleNamespace(
                source="https://arxiv.org/abs/2602.10094",
                title="arxiv.org — 4RC",
                text="Paper abstract.",
            )
        )
        return "Ingested 1 document(s): 4RC."

    async def fake_rag_search(**kwargs):
        assert kwargs["docs"] == ["https://arxiv.org/abs/2602.10094"]
        return []

    monkeypatch.setattr(
        web_agent,
        "observable_run_with_manual_validation_retries",
        fake_model_run,
    )
    monkeypatch.setattr(web_agent, "web_crawl_and_ingest", fake_crawl)
    monkeypatch.setattr(web_agent, "rag_search_documents", fake_rag_search)
    monkeypatch.setattr(
        web_agent,
        "_save_arxiv_documents",
        lambda documents: (
            ["/docs/papers/arxiv/2602.10094.md"] if documents else []
        ),
    )

    result = await web_agent.run_web_task(
        "fetch https://arxiv.org/html/2602.10094 for me"
    )

    assert crawls == [
        "https://arxiv.org/html/2602.10094",
        "https://arxiv.org/abs/2602.10094",
    ]
    assert "Saved locally: `/docs/papers/arxiv/2602.10094.md`" in result


def test_current_turn_prompt_prioritizes_user_request():
    from run_agents import _current_turn_prompt

    prompt = _current_turn_prompt("modify the title in the reply")

    assert prompt.startswith("## Current User Request")
    assert "authoritative instruction" in prompt
    assert "modify the title" in prompt
    assert "/skills/fitness/diet.md" not in prompt


def test_recent_orchestrator_history_is_bounded():
    from run_agents import MAX_ORCHESTRATOR_HISTORY_MESSAGES, _recent_orchestrator_history

    messages = [
        ModelRequest.user_text_prompt(f"message {idx}")
        for idx in range(MAX_ORCHESTRATOR_HISTORY_MESSAGES + 3)
    ]

    assert _recent_orchestrator_history(messages) == messages[
        -MAX_ORCHESTRATOR_HISTORY_MESSAGES:
    ]


def test_clean_text_answer_trims_small_model_self_review():
    from agents.structured_retry import clean_text_answer

    answer = (
        "The local collection contains five papers. Each paper was summarized "
        "from its assigned files.\n\n"
        "Wait, I need to ensure I followed every instruction.\n"
        "Final check: repeat the prompt."
    )

    assert clean_text_answer(answer) == (
        "The local collection contains five papers. Each paper was summarized "
        "from its assigned files."
    )


def test_clean_text_answer_keeps_short_user_facing_wait_sentence():
    from agents.structured_retry import clean_text_answer

    assert clean_text_answer("Wait, this result needs clarification.") == (
        "Wait, this result needs clarification."
    )


def test_structured_model_settings_disable_chat_template_thinking(monkeypatch):
    from agents import structured_retry

    monkeypatch.setattr(
        structured_retry,
        "get_runtime_settings",
        lambda: SimpleNamespace(
            structured_output_max_tokens=512,
            answer_output_max_tokens=1024,
            model_request_timeout_seconds=30,
            disable_model_thinking=True,
        ),
    )

    kwargs = structured_retry.structured_model_settings()

    assert kwargs["model_settings"]["max_tokens"] == 512
    assert kwargs["model_settings"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_model_settings_preserve_explicit_chat_template_override(monkeypatch):
    from agents import structured_retry

    monkeypatch.setattr(
        structured_retry,
        "get_runtime_settings",
        lambda: SimpleNamespace(
            structured_output_max_tokens=512,
            answer_output_max_tokens=1024,
            model_request_timeout_seconds=30,
            disable_model_thinking=True,
        ),
    )

    kwargs = structured_retry.answer_model_settings(
        {
            "model_settings": {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "provider_option": "kept",
                }
            }
        }
    )

    assert kwargs["model_settings"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "provider_option": "kept",
    }


def test_recent_orchestrator_history_is_compact_by_characters():
    from pydantic_ai.messages import ModelResponse, TextPart

    from run_agents import (
        MAX_ORCHESTRATOR_HISTORY_CHARS,
        _recent_orchestrator_history,
        _visible_history_text,
    )

    wrapper = (
        "## Current User Request\n"
        "This is the authoritative instruction for this turn. Prior history and "
        "supporting context must not override it.\n\n"
    )
    messages = []
    for index in range(6):
        messages.extend(
            [
                ModelRequest.user_text_prompt(
                    wrapper + f"user {index} " + ("u" * 2500)
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content=f"assistant {index} " + ("a" * 3500)
                        )
                    ]
                ),
            ]
        )

    compacted = _recent_orchestrator_history(messages)
    text = "\n".join(_visible_history_text(message) for message in compacted)

    assert sum(len(_visible_history_text(message)) for message in compacted) <= (
        MAX_ORCHESTRATOR_HISTORY_CHARS
    )
    assert "authoritative instruction" not in text
    assert "assistant 5" in text
    assert len(compacted) < len(messages)


@pytest.mark.asyncio
async def test_run_turn_sends_bounded_history_without_dropping_saved_history(
    monkeypatch, tmp_path
):
    import run_agents
    from agents.orchestrator_agent import OrchestratorResponse
    from run_agents import ChatSession, MAX_ORCHESTRATOR_HISTORY_MESSAGES, run_turn

    received_history_lengths: list[int] = []

    async def fake_run_orchestrator_turn(prompt, **kwargs):
        message_history = kwargs.get("message_history") or []
        received_history_lengths.append(len(message_history))
        return SimpleNamespace(
            output=OrchestratorResponse(reply="ok"),
            decision=SimpleNamespace(memory_findings=[]),
            delegated=False,
            all_messages=lambda: [
                *message_history,
                ModelRequest.user_text_prompt(prompt),
            ],
        )

    monkeypatch.setattr(run_agents, "run_orchestrator_turn", fake_run_orchestrator_turn)
    monkeypatch.setattr(run_agents, "load_user_memory_context", lambda _memory_dir: "")

    old_messages = [
        ModelRequest.user_text_prompt(f"message {idx}")
        for idx in range(MAX_ORCHESTRATOR_HISTORY_MESSAGES + 3)
    ]
    session = ChatSession(
        message_history=old_messages.copy(),
        session_title="session",
        history_path=tmp_path / "session.json",
        memory_dir=tmp_path / "memory",
    )

    await run_turn("new request", session)

    assert received_history_lengths == [MAX_ORCHESTRATOR_HISTORY_MESSAGES]
    assert session.message_history[: len(old_messages)] == old_messages
    assert len(session.message_history) == len(old_messages) + 1


def test_orchestrator_choice_is_normalized_by_python():
    from agents.orchestrator_agent import (
        OrchestratorChoice,
        _decision_from_choice,
    )

    assert set(OrchestratorChoice.model_json_schema()["properties"]) == {
        "route",
        "content",
    }

    direct = _decision_from_choice(
        "hello",
        OrchestratorChoice(route="direct", content="Hello."),
    )
    assert direct.reply == "Hello."
    assert direct.objective is None
    assert direct.effort == "none"

    delegated = _decision_from_choice(
        "fallback objective",
        OrchestratorChoice(route="web", content="Fetch the current docs."),
    )
    assert delegated.reply is None
    assert delegated.objective == "Fetch the current docs."
    assert delegated.effort == "minimal"

    planned = _decision_from_choice(
        "Compare local notes with current sources.",
        OrchestratorChoice(route="plan"),
    )
    assert planned.objective == "Compare local notes with current sources."
    assert planned.effort == "standard"


def test_orchestrator_drops_invented_fs_mount_path():
    from agents.orchestrator_agent import (
        OrchestratorChoice,
        _decision_from_choice,
    )

    prompt = "check the papers locally related to world models and summarize them"
    decision = _decision_from_choice(
        prompt,
        OrchestratorChoice(
            route="fs",
            content=(
                "I will read local papers under /skills and summarize them."
            ),
        ),
    )

    assert decision.objective == prompt


def test_agent_runtime_settings_read_dotenv(monkeypatch, tmp_path):
    from localagent_settings import get_runtime_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCALAGENT_MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("LOCALAGENT_SKILLS_MODE", raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LOCALAGENT_MEMORY_ENABLED=false",
                "LOCALAGENT_SKILLS_MODE=RO",
            ]
        ),
        encoding="utf-8",
    )
    get_runtime_settings.cache_clear()

    settings = get_runtime_settings()

    assert settings.memory_enabled is False
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
        "LOCALAGENT_AGENT_TURN_TIMEOUT_SECONDS",
    }
    compose_keys = {"LOCALAGENT_BIND", "LOCALAGENT_PORT"}

    supported = runtime_keys | speech_keys | server_keys | compose_keys

    assert "OPENAI_API_KEY" not in env_keys
    assert "ANTHROPIC_API_KEY" not in env_keys
    assert env_keys <= supported
    assert runtime_keys <= env_keys
    assert speech_keys <= env_keys


def test_orchestrator_prompt_declares_validator_mount_access():
    from agents.orchestrator_agent import _orchestrator_prompt_body

    prompt = _orchestrator_prompt_body()

    assert "Filesystem access contract:" in prompt
    assert "/docs" in prompt
    assert "/skills" in prompt
    assert "choose fs or plan" in prompt
    assert "Do not answer that you lack access" in prompt


def test_orchestrator_prompt_declares_fast_specialist_routes():
    from agents.orchestrator_agent import _orchestrator_prompt_body

    prompt = _orchestrator_prompt_body()
    normalized = " ".join(prompt.split())

    assert "Prefer making progress:" in prompt
    assert "Choose direct when" in prompt
    assert "Choose fs for one narrow filesystem task" in prompt
    assert "Choose web for one narrow current/web task" in prompt
    assert "Choose plan when" in prompt
    assert "There is no clarify route" in prompt
    assert "Classify by the information source and missing user intent" in prompt
    assert "Stable conceptual" in prompt
    assert "not by topic keywords" in normalized
    assert "not executing the route" in prompt
    assert "Return only route and content." in prompt
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
async def test_orchestrator_fs_decision_forwards_specialist_answer(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[str] = []

    async def fake_fs_task(objective: str) -> SpecialistResult:
        calls.append(objective)
        return SpecialistResult(
            agent="fs_agent",
            answer="The file says hello.",
            summary="Read one file",
        )

    async def fake_plan_workflow(*_args, **_kwargs):
        raise AssertionError("fs fast route must not call plan workflow")

    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_plan_workflow", fake_plan_workflow)

    response, messages = await _response_and_messages(
        OrchestratorDecision(route="fs", objective="read /docs/a.md"),
        [],
    )

    assert response.reply == "The file says hello."
    assert messages == []
    assert calls == ["read /docs/a.md"]


@pytest.mark.asyncio
async def test_orchestrator_turn_guardrail_prevents_recent_research_fs_run(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorChoice, run_orchestrator_turn
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[str] = []

    async def fake_decision(*_args, **_kwargs):
        return SimpleNamespace(
            output=OrchestratorChoice(
                route="fs",
                content="Provide recent language model research overview",
            ),
            all_messages=lambda: [],
        )

    async def fail_fs_task(_objective: str):
        raise AssertionError("guardrail should prevent fs_agent from running")

    async def fake_web_task(objective: str) -> SpecialistResult:
        calls.append(objective)
        return SpecialistResult(
            agent="web_agent",
            answer="Recent language model research from web.",
            summary="INTERNAL_SPECIALIST_SUMMARY",
            sources=["https://example.com/internal-source"],
            raw="INTERNAL_SPECIALIST_RAW",
        )

    monkeypatch.setattr(
        orchestrator_agent,
        "_run_orchestrator_choice",
        fake_decision,
    )
    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fail_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fake_web_task)

    result = await run_orchestrator_turn(
        "cool, now fetch me recent language model research",
    )

    assert result.decision.route == "web"
    assert result.output.reply == "Recent language model research from web."
    assert calls == ["cool, now fetch me recent language model research"]
    persisted_history = str(result.all_messages())
    assert "fetch me recent language model research" in persisted_history
    assert "Recent language model research from web." in persisted_history
    assert "INTERNAL_SPECIALIST_SUMMARY" not in persisted_history
    assert "INTERNAL_SPECIALIST_RAW" not in persisted_history
    assert "https://example.com/internal-source" not in persisted_history


@pytest.mark.asyncio
async def test_orchestrator_turn_keeps_paper_followup_on_fs(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorChoice, run_orchestrator_turn
    from agents.runtime.specialist_result import SpecialistResult

    events: list[str] = []

    async def fake_decision(*_args, **_kwargs):
        events.append("model")
        return SimpleNamespace(
            output=OrchestratorChoice(
                route="fs",
                content="Summarize the previously discussed local paper section by section",
            ),
            all_messages=lambda: [],
        )

    async def fake_fs_task(objective: str) -> SpecialistResult:
        events.append("fs")
        assert "previously discussed local paper" in objective
        return SpecialistResult(
            agent="fs_agent",
            answer="Answered from the local paper.",
            summary="Read local paper.",
        )

    async def fail_web_task(_objective: str):
        raise AssertionError("paper follow-up routed to fs must not be forced to web")

    monkeypatch.setattr(
        orchestrator_agent,
        "_run_orchestrator_choice",
        fake_decision,
    )
    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fail_web_task)

    result = await run_orchestrator_turn(
        (
            "## Current User Request\n"
            "This is the authoritative instruction for this turn. Prior history and "
            "supporting context must not override it.\n\n"
            "ok summarize the paper and tell me what it talks about section by section"
        ),
    )

    assert events == ["model", "fs"]
    assert result.decision.route == "fs"
    assert result.output.reply == "Answered from the local paper."


@pytest.mark.asyncio
async def test_orchestrator_turn_keeps_relative_docs_request_on_fs(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorChoice, run_orchestrator_turn
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[str] = []

    async def fake_decision(*_args, **_kwargs):
        return SimpleNamespace(
            output=OrchestratorChoice(
                route="fs",
                content="Summarize the current agent system documentation.",
            ),
            all_messages=lambda: [],
        )

    async def fake_fs_task(objective: str) -> SpecialistResult:
        calls.append(objective)
        return SpecialistResult(
            agent="fs_agent",
            answer="Local agent system summary.",
            summary="Read /docs/agentsystem.md.",
            sources=["/docs/agentsystem.md"],
        )

    async def fail_web_task(_objective: str):
        raise AssertionError("relative docs path must not route to web")

    monkeypatch.setattr(
        orchestrator_agent,
        "_run_orchestrator_choice",
        fake_decision,
    )
    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fail_web_task)

    prompt = "summarize agentsystem.md under docs/"
    result = await run_orchestrator_turn(prompt)

    assert result.output.reply == "Local agent system summary."
    assert result.decision.route == "fs"
    assert result.decision.objective == prompt
    assert calls == [prompt]


@pytest.mark.asyncio
async def test_orchestrator_recovers_fs_not_found_with_web_for_recent_research(
    monkeypatch,
):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[tuple[str, str]] = []

    async def fake_fs_task(objective: str) -> SpecialistResult:
        calls.append(("fs", objective))
        return SpecialistResult(
            agent="fs_agent",
            status="not_found",
            useful=False,
            recoverable_by_web=True,
            answer="I could not find local language model research notes.",
            summary="Local reference unavailable.",
            uncertainties=["No matching local docs were found."],
        )

    async def fake_web_task(objective: str) -> SpecialistResult:
        calls.append(("web", objective))
        return SpecialistResult(
            agent="web_agent",
            answer="Recent language model research recovered from web.",
            summary="Recovered externally.",
            sources=["https://arxiv.org/abs/2605.06548"],
        )

    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fake_web_task)

    response, messages = await _response_and_messages(
        OrchestratorDecision(
            route="fs",
            objective="Provide recent language model research overview",
        ),
        [],
        original_prompt="fetch me recent language model research",
    )

    assert response.reply == "Recent language model research recovered from web."
    assert messages == []
    assert calls[0] == ("fs", "Provide recent language model research overview")
    assert calls[1][0] == "web"
    assert "Local filesystem lookup failed" in calls[1][1]


@pytest.mark.asyncio
async def test_orchestrator_recovers_ambiguous_local_lookup_with_web(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[tuple[str, str]] = []

    async def fake_fs_task(objective: str) -> SpecialistResult:
        calls.append(("fs", objective))
        return SpecialistResult(
            agent="fs_agent",
            status="not_found",
            useful=False,
            recoverable_by_web=True,
            answer="I could not find a relevant local paper.",
            summary="No useful local paper matched.",
        )

    async def fake_web_task(objective: str) -> SpecialistResult:
        calls.append(("web", objective))
        return SpecialistResult(
            agent="web_agent",
            answer="The paper was recovered from the web.",
            summary="Found external source.",
        )

    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fake_web_task)

    response, _messages = await _response_and_messages(
        OrchestratorDecision(
            route="fs",
            objective="summarize the paper and explain its architecture",
        ),
        [],
        original_prompt="summarize the paper and explain its architecture",
    )

    assert response.reply == "The paper was recovered from the web."
    assert [kind for kind, _objective in calls] == ["fs", "web"]


@pytest.mark.asyncio
async def test_orchestrator_recovers_local_paper_search_with_web(
    monkeypatch,
):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[tuple[str, str]] = []

    async def fake_fs_task(objective: str) -> SpecialistResult:
        calls.append(("fs", objective))
        return SpecialistResult(
            agent="fs_agent",
            status="not_found",
            useful=False,
            recoverable_by_web=True,
            answer="I could not find matching local papers.",
            summary="No matching local papers.",
        )

    async def fake_web_task(objective: str) -> SpecialistResult:
        calls.append(("web", objective))
        return SpecialistResult(
            agent="web_agent",
            answer="Recovered matching papers from the web.",
            summary="External paper recovery completed.",
        )

    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fake_web_task)

    response, _messages = await _response_and_messages(
        OrchestratorDecision(
            route="fs",
            objective="check local papers regarding recent world model research",
        ),
        [],
        original_prompt="check local papers regarding recent world model research",
    )

    assert response.reply == "Recovered matching papers from the web."
    assert [kind for kind, _objective in calls] == ["fs", "web"]


@pytest.mark.asyncio
async def test_orchestrator_does_not_recover_explicit_docs_miss_with_web(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[str] = []

    async def fake_fs_task(objective: str) -> SpecialistResult:
        calls.append(objective)
        return SpecialistResult(
            agent="fs_agent",
            status="not_found",
            useful=False,
            recoverable_by_web=True,
            answer="I could not find /docs/missing.md.",
            summary="Explicit local path unavailable.",
            uncertainties=["Invalid path hint(s): /docs/missing.md."],
        )

    async def fail_web_task(_objective: str) -> SpecialistResult:
        raise AssertionError("explicit local path miss should not recover to web")

    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fail_web_task)

    response, messages = await _response_and_messages(
        OrchestratorDecision(route="fs", objective="Read /docs/missing.md"),
        [],
        original_prompt="read /docs/missing.md",
    )

    assert response.reply == "I could not find /docs/missing.md."
    assert messages == []
    assert calls == ["Read /docs/missing.md"]


@pytest.mark.asyncio
async def test_orchestrator_recovers_inferred_local_match_with_web(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[tuple[str, str]] = []

    async def fake_fs_task(objective: str) -> SpecialistResult:
        calls.append(("fs", objective))
        return SpecialistResult(
            agent="fs_agent",
            status="not_found",
            useful=False,
            recoverable_by_web=True,
            answer="The inferred local paper path was unavailable.",
            summary="Local arXiv paper not found.",
        )

    async def fake_web_task(objective: str) -> SpecialistResult:
        calls.append(("web", objective))
        return SpecialistResult(
            agent="web_agent",
            answer="Recovered the paper from arXiv.",
            summary="Fetched external paper.",
        )

    monkeypatch.setattr(orchestrator_agent, "_run_fs_task_result", fake_fs_task)
    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fake_web_task)

    response, _messages = await _response_and_messages(
        OrchestratorDecision(
            route="fs",
            objective="Locate local paper 2605.00080v1 and summarize it",
        ),
        [],
        original_prompt="what about 2605.00080v1?",
    )

    assert response.reply == "Recovered the paper from arXiv."
    assert [kind for kind, _objective in calls] == ["fs", "web"]


@pytest.mark.asyncio
async def test_orchestrator_web_decision_forwards_specialist_answer(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import OrchestratorDecision, _response_and_messages
    from agents.runtime.specialist_result import SpecialistResult

    calls: list[str] = []

    async def fake_web_task(objective: str) -> SpecialistResult:
        calls.append(objective)
        return SpecialistResult(
            agent="web_agent",
            answer="The current docs say to use v2.",
            summary="Read current docs.",
            sources=["https://example.com/docs"],
        )

    async def fake_plan_workflow(*_args, **_kwargs):
        raise AssertionError("web fast route must not call plan workflow")

    monkeypatch.setattr(orchestrator_agent, "_run_web_task_result", fake_web_task)
    monkeypatch.setattr(orchestrator_agent, "_run_plan_workflow", fake_plan_workflow)

    response, messages = await _response_and_messages(
        OrchestratorDecision(route="web", objective="read https://example.com/docs"),
        [],
    )

    assert response.reply == "The current docs say to use v2."
    assert messages == []
    assert calls == ["read https://example.com/docs"]


@pytest.mark.asyncio
async def test_orchestrator_plan_decision_uses_effort_budget(monkeypatch):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import (
        OrchestratorDecision,
        _response_and_messages,
    )

    calls: list[tuple[str, int, int]] = []

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
            "Orchestrator notes:\n- Findings available: 1"
        )

    monkeypatch.setattr(orchestrator_agent, "_run_plan_workflow", fake_plan_workflow)

    response, messages = await _response_and_messages(
        OrchestratorDecision(
            route="plan",
            objective="read notes.md",
            effort="minimal",
        ),
        [],
    )

    assert response.reply == "Read result for read notes.md."
    assert messages == []
    assert response.session_title is None
    assert calls == [("read notes.md", 1, 1)]


@pytest.mark.asyncio
async def test_orchestrator_plan_answer_falls_back_to_forwardable_research(
    monkeypatch,
):
    from agents import orchestrator_agent
    from agents.orchestrator_agent import (
        OrchestratorDecision,
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

    monkeypatch.setattr(orchestrator_agent, "_run_plan_workflow", fake_plan_workflow)

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


@pytest.mark.asyncio
async def test_orchestrator_collection_preflight_skips_route_model(monkeypatch):
    from agents import orchestrator_agent

    async def fail_route_model(*_args, **_kwargs):
        raise AssertionError("collection preflight must skip the route model")

    async def fake_plan_workflow(
        objective: str,
        *,
        max_tasks: int,
        max_iterations: int,
    ) -> str:
        assert objective == "now summarize all papers locally in parallel"
        assert (max_tasks, max_iterations) == (3, 2)
        return "Forwardable answer:\nAll local papers were summarized."

    monkeypatch.setattr(
        orchestrator_agent,
        "_run_orchestrator_choice",
        fail_route_model,
    )
    monkeypatch.setattr(
        orchestrator_agent,
        "_run_plan_workflow",
        fake_plan_workflow,
    )

    result = await orchestrator_agent.run_orchestrator_turn(
        "now summarize all papers locally in parallel"
    )

    assert result.decision.route == "plan"
    assert result.decision.objective == "now summarize all papers locally in parallel"
    assert result.output.reply == "All local papers were summarized."


@pytest.mark.asyncio
async def test_orchestrator_plan_route_persists_only_visible_turn(monkeypatch):
    from agents import orchestrator_agent
    from agents import structured_retry
    from agents.orchestrator_agent import OrchestratorChoice, run_orchestrator_turn

    previous = ModelRequest.user_text_prompt("previous")

    async def fake_observable_run(_agent, prompt, **kwargs):
        assert kwargs.get("message_history") == [previous]
        return SimpleNamespace(
            output=OrchestratorChoice(
                route="plan",
                content="read notes",
            ),
            all_messages=lambda: [
                *kwargs.get("message_history"),
                ModelRequest.user_text_prompt(prompt),
            ],
        )

    async def fake_plan_workflow(*_args, **_kwargs):
        return (
            "Forwardable answer:\n"
            "Visible final answer.\n\n"
            "Orchestrator notes:\n"
            "- internal note"
        )

    monkeypatch.setattr(structured_retry, "observable_run", fake_observable_run)
    monkeypatch.setattr(orchestrator_agent, "_run_plan_workflow", fake_plan_workflow)

    result = await run_orchestrator_turn(
        "current prompt",
        message_history=[previous],
    )

    assert result.output.reply == "Visible final answer."
    assert len(result.all_messages()) == 3
    assert result.all_messages()[0] == previous
    assert "current prompt" in str(result.all_messages()[1])
    assert "Visible final answer." in str(result.all_messages()[2])
    assert "Orchestrator notes" not in str(result.all_messages())


@pytest.mark.asyncio
async def test_manual_structured_retry_does_not_replay_invalid_output(monkeypatch):
    from agents import structured_retry
    from agents.orchestrator_agent import OrchestratorDecision
    from agents.structured_retry import observable_run_with_manual_validation_retries
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    previous = ModelRequest.user_text_prompt("previous")
    prompts: list[str] = []
    histories: list[list[object] | None] = []

    async def fake_observable_run(_agent, prompt, **kwargs):
        prompts.append(prompt)
        histories.append(kwargs.get("message_history"))
        if len(prompts) == 1:
            raise UnexpectedModelBehavior(
                "Exceeded maximum retries (0) for output validation"
            )
        return SimpleNamespace(
            output=OrchestratorDecision(route="direct", reply="ok"),
            all_messages=lambda: [previous, ModelRequest.user_text_prompt(prompt)],
        )

    monkeypatch.setattr(structured_retry, "observable_run", fake_observable_run)

    result = await observable_run_with_manual_validation_retries(
        object(),
        "current prompt",
        output_type=OrchestratorDecision,
        output_name="OrchestratorDecision",
        label="orchestrator",
        message_history=[previous],
        attempts=2,
    )

    assert result.output.reply == "ok"
    assert histories == [[previous], [previous]]
    assert prompts[0] == "current prompt"
    assert "current prompt" in prompts[1]
    assert "invalid response is intentionally omitted" in prompts[1]
    assert "Previous invalid output:" not in prompts[1]


def test_agent_output_validation_retries_are_disabled():
    from agents.fs_agent import fs_agent
    from agents.orchestrator_agent import orchestrator
    from agents.plan_agent import plan_agent
    from agents.web_agent import web_answer_agent

    assert orchestrator._max_result_retries == 0
    assert plan_agent._max_result_retries == 0
    assert fs_agent._max_result_retries == 0
    assert web_answer_agent._max_result_retries == 0


def test_save_history_includes_memory_dir(tmp_path):
    from run_agents import ChatSession, _save_history

    history_path = tmp_path / "chat.json"
    memory_dir = tmp_path / "memory" / "default"
    session = ChatSession(
        message_history=[ModelRequest.user_text_prompt("hello")],
        session_title="session",
        history_path=history_path,
        memory_dir=memory_dir,
    )

    _save_history(session)

    payload = json.loads(history_path.read_text())
    assert payload["session_title"] == "session"
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


def test_new_session_sets_history_path(monkeypatch, tmp_path):
    import run_agents
    from run_agents import ChatSession, _init_session_paths_from_user_text

    monkeypatch.setattr(run_agents, "CHAT_HISTORY_DIR", tmp_path / "chats")

    session = ChatSession()
    _init_session_paths_from_user_text(session, "read notes")

    assert session.session_title == "read-notes"
    assert session.history_path == tmp_path / "chats" / "read-notes.json"


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

    session = ChatSession(
        session_title="session",
        history_path=tmp_path / "session.json",
        memory_dir=memory_dir,
    )

    await run_turn("first", session)
    await run_turn("second", session)

    assert "private deployment key" not in calls[0][0]
    assert "private deployment key" in calls[0][1]
    assert "private deployment key" not in calls[1][0]
    assert calls[1][1] == ""


def test_new_session_uses_unique_history_stem(monkeypatch, tmp_path):
    import run_agents
    from run_agents import ChatSession, _init_session_paths_from_user_text

    monkeypatch.setattr(run_agents, "CHAT_HISTORY_DIR", tmp_path / "chats")
    existing_history = tmp_path / "chats" / "read-notes.json"
    existing_history.parent.mkdir(parents=True)
    existing_history.write_text("{}", encoding="utf-8")

    session = ChatSession()
    _init_session_paths_from_user_text(session, "read notes")

    assert session.session_title == "read-notes-2"
    assert session.history_path == tmp_path / "chats" / "read-notes-2.json"


def test_skills_context_includes_current_skill_paths():
    from agents.runtime.skills_context import scan_skills_context

    context = scan_skills_context()

    assert "Current /skills catalog" in context
    assert "fitness/diet.md" in context
    assert "fitness/workout.md" in context


def test_fs_task_prompt_routes_missing_file_to_recovery(monkeypatch, tmp_path):
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

    prompt, analysis = fs_agent._fs_task_prompt("read /docs/not-present.md")

    assert analysis.invalid_paths == ["/docs/not-present.md"]
    assert "Missing-path chain:" in prompt
    assert "find_paths on the task scope (/docs)" in prompt
    assert "-> grep_files if content can identify the file" in prompt
    assert "Ask for confirmation before edits" in prompt


def test_fs_task_prompt_includes_candidate_paths_for_confirmation(monkeypatch, tmp_path):
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

    prompt, analysis = fs_agent._fs_task_prompt("read /docs/acutal.md")

    assert analysis.invalid_paths == ["/docs/acutal.md"]
    assert analysis.candidate_paths == ["/docs/actual.md"]
    assert "Replacement candidates" in prompt
    assert "- /docs/actual.md" in prompt


def test_fs_task_prompt_finds_cross_root_candidate_for_missing_writable_path(
    monkeypatch,
    tmp_path,
):
    from agents import fs_agent

    docs = tmp_path / "docs"
    skills = tmp_path / "skills"
    docs.mkdir()
    skills.mkdir()
    (docs / "agentsystem.md").write_text("hello", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[
                Mount(host_path=docs, mount_point="/docs", mode="ro"),
                Mount(host_path=skills, mount_point="/skills", mode="rw"),
            ]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    prompt, analysis = fs_agent._fs_task_prompt("summarize /skills/agentsystem.md")

    assert analysis.invalid_paths == ["/skills/agentsystem.md"]
    assert analysis.write_targets == ["/skills/agentsystem.md"]
    assert analysis.candidate_paths == ["/docs/agentsystem.md"]
    assert "New write targets" in prompt
    assert "- /skills/agentsystem.md" in prompt
    assert "Replacement candidates" in prompt
    assert "- /docs/agentsystem.md" in prompt
    assert "use one clear replacement candidate for a read-only request" in prompt


def test_fs_task_prompt_tells_agent_to_use_resolved_paths_first(monkeypatch, tmp_path):
    from agents import fs_agent

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "actual.md").write_text("hello", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    prompt, analysis = fs_agent._fs_task_prompt("read /docs/actual.md")

    assert analysis.resolved_paths == ["/docs/actual.md"]
    assert "Mode: exact_path" in prompt
    assert "find_paths" not in prompt


def test_fs_path_recovery_guard_allows_read_only_candidate_answer(
    monkeypatch,
    tmp_path,
):
    from agents import fs_agent
    from agents.fs_agent import (
        FsAgentResult,
        PathAnalysis,
        _apply_path_recovery_guard,
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "actual.md").write_text("hello", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    output = FsAgentResult(
        answer="The file says hello.",
        summary="Found a possible file.",
        paths=["/docs/actual.md"],
    )
    guarded = _apply_path_recovery_guard(
        output,
        PathAnalysis(
            invalid_paths=["/skills/agentsystem.md"],
            candidate_paths=["/docs/actual.md"],
        ),
    )

    assert "Answered from likely replacement" in guarded.uncertainties[0]
    assert "Heads up:" in (guarded.answer or "")
    assert "could not find /skills/agentsystem.md" in (guarded.answer or "")
    assert "/docs/actual.md is probably the closest match" in (guarded.answer or "")
    assert "The file says hello." in (guarded.answer or "")


def test_fs_path_recovery_guard_preserves_valid_path_answer(monkeypatch, tmp_path):
    from agents import fs_agent
    from agents.fs_agent import (
        FsAgentResult,
        PathAnalysis,
        _apply_path_recovery_guard,
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("valid", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    output = FsAgentResult(
        answer="Summary of /docs/a.md.",
        summary="Read valid path.",
        paths=["/docs/a.md"],
    )
    guarded = _apply_path_recovery_guard(
        output,
        PathAnalysis(
            resolved_paths=["/docs/a.md"],
            invalid_paths=["/docs/missing.md"],
            candidate_paths=["/docs/a.md"],
        ),
    )

    assert "Summary of /docs/a.md." in (guarded.answer or "")
    assert "/docs/a.md is probably the closest match" not in (guarded.answer or "")
    assert "Handled valid requested path(s): /docs/a.md" in guarded.uncertainties[0]


def test_fs_rag_paths_preserve_valid_paths_with_invalid_hint(monkeypatch, tmp_path):
    from agents import fs_agent
    from agents.fs_agent import FsAgentResult, PathAnalysis, _rag_paths_for_output

    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    (docs / "large.md").write_text("valid", encoding="utf-8")
    (nested / "child.md").write_text("child", encoding="utf-8")
    (docs / "candidate.md").write_text("candidate", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    paths = _rag_paths_for_output(
        FsAgentResult(
            answer="Summary.",
            summary="Read valid path.",
            paths=[
                "/docs/large.md",
                "/docs/nested/child.md",
                "/docs/candidate.md",
            ],
        ),
        PathAnalysis(
            resolved_paths=["/docs/large.md", "/docs/nested"],
            invalid_paths=["/docs/missing.md"],
            candidate_paths=["/docs/candidate.md"],
        ),
    )

    assert paths == ["/docs/large.md", "/docs/nested", "/docs/nested/child.md"]


@pytest.mark.asyncio
async def test_fs_ambiguous_local_paper_uses_filesystem_discovery(
    monkeypatch,
    tmp_path,
):
    from agents import fs_agent

    docs = tmp_path / "docs"
    papers = docs / "papers" / "arxiv"
    papers.mkdir(parents=True)
    (papers / "world-model.md").write_text(
        "# World Model Survey\nRelevant evidence.",
        encoding="utf-8",
    )
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    prompts: list[str] = []

    retrieved_paths: list[str] = []

    async def fake_rag(objective, paths):
        assert "world model" in objective
        retrieved_paths.extend(paths)
        return [
            {
                "node_id": "world-model-node",
                "source": paths[0],
                "title": "World Model Survey",
                "text": "The paper explains predictive dynamics.",
            }
        ]

    async def fake_synthesis(**_kwargs):
        return "The selected local World Model Survey explains predictive dynamics."

    async def fake_fs_tools(
        prompt,
        *,
        question,
        task_roots,
        discovery_preview_only=False,
        discovery_search_paths=None,
    ):
        prompts.append(prompt)
        assert question == (
            "check the papers locally related to world model, and summarize it"
        )
        assert task_roots == ["/docs"]
        assert discovery_preview_only is True
        assert discovery_search_paths == ["/docs"]
        return (
            "The World Model Survey is the strongest lexical candidate.",
            [
                (
                    "grep_files",
                    {"query": "world model", "path": "/docs"},
                ),
                (
                    "preview_file",
                    {"path": "/docs/papers/arxiv/world-model.md"},
                ),
            ],
        )

    monkeypatch.setattr(fs_agent, "_retrieve_rag_evidence", fake_rag)
    monkeypatch.setattr(fs_agent, "_synthesize_rag_answer", fake_synthesis)
    monkeypatch.setattr(fs_agent, "_run_fs_agent", fake_fs_tools)

    result = await fs_agent.run_fs_task_result(
        "check the papers locally related to world model, and summarize it"
    )

    assert result.status == "ok"
    assert "predictive dynamics" in result.answer
    assert "/docs/papers/arxiv/world-model.md" in result.sources
    assert retrieved_paths == ["/docs/papers/arxiv/world-model.md"]
    assert len(prompts) == 1
    assert "Mode: topic_discovery" in prompts[0]
    assert "Search path: /docs" in prompts[0]


@pytest.mark.asyncio
async def test_fs_non_paper_docs_use_filesystem_instead_of_papers_rag(
    monkeypatch,
    tmp_path,
):
    from agents import fs_agent

    docs = tmp_path / "docs"
    papers = docs / "papers"
    papers.mkdir(parents=True)
    (docs / "agentsystem.md").write_text("Agent system notes.", encoding="utf-8")
    (papers / "world-model.md").write_text("World model paper.", encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    retrieved_paths: list[str] = []

    async def fake_rag(objective, paths):
        assert "agent system" in objective
        retrieved_paths.extend(paths)
        return [
            {
                "node_id": "agent-system-node",
                "source": paths[0],
                "title": "Agent System",
                "text": "The documentation describes scoped routing.",
            }
        ]

    async def fake_synthesis(**_kwargs):
        return "The local agent system documentation describes scoped routing."

    async def fake_fs_tools(
        prompt,
        *,
        question,
        task_roots,
        discovery_preview_only=False,
        discovery_search_paths=None,
    ):
        assert question == "summarize the local agent system documentation"
        assert task_roots == ["/docs"]
        assert discovery_preview_only is True
        assert discovery_search_paths == ["/docs"]
        assert "Mode: topic_discovery" in prompt
        assert "Search path: /docs" in prompt
        assert "Readable file index" not in prompt
        return (
            "The agent system document is the strongest candidate.",
            [
                (
                    "grep_files",
                    {"query": "agent system", "path": "/docs"},
                ),
                (
                    "preview_file",
                    {"path": "/docs/agentsystem.md"},
                ),
            ],
        )

    monkeypatch.setattr(fs_agent, "_retrieve_rag_evidence", fake_rag)
    monkeypatch.setattr(fs_agent, "_synthesize_rag_answer", fake_synthesis)
    monkeypatch.setattr(fs_agent, "_run_fs_agent", fake_fs_tools)

    result = await fs_agent.run_fs_task_result(
        "summarize the local agent system documentation"
    )

    assert result.status == "ok"
    assert result.sources == ["/docs/agentsystem.md"]
    assert retrieved_paths == ["/docs/agentsystem.md"]
    assert "scoped routing" in result.answer


def test_fs_result_metadata_is_assembled_from_tool_calls():
    from agents.fs_agent import _build_fs_output

    output = _build_fs_output(
        answer="Updated the skill.",
        paths=[],
        calls=[
            ("read_file", {"path": "/skills/research/strategy.md"}),
            ("edit_file", {"path": "/skills/research/strategy.md"}),
        ],
    )

    assert output.paths == ["/skills/research/strategy.md"]
    assert output.changes_made == [
        "edit_file: /skills/research/strategy.md"
    ]


def test_fs_path_recovery_guard_requires_confirmation_without_candidate_read():
    from agents.fs_agent import (
        FsAgentResult,
        PathAnalysis,
        _apply_path_recovery_guard,
    )

    output = FsAgentResult(
        answer="I found this file.",
        summary="Found a possible file.",
        paths=[],
    )
    guarded = _apply_path_recovery_guard(
        output,
        PathAnalysis(
            invalid_paths=["/skills/agentsystem.md"],
            candidate_paths=["/docs/actual.md"],
        ),
    )

    assert "Exact-path confirmation is required" in guarded.uncertainties[0]
    assert "/docs/actual.md" in guarded.uncertainties[0]
    assert "Please confirm the exact path" in (guarded.answer or "")


def test_fs_path_recovery_guard_reports_not_found_without_candidates():
    from agents.fs_agent import (
        FsAgentResult,
        PathAnalysis,
        _apply_path_recovery_guard,
    )

    output = FsAgentResult(answer=None, summary="No result.")
    guarded = _apply_path_recovery_guard(
        output,
        PathAnalysis(invalid_paths=["/skills/agentsystem.md"]),
    )

    assert "could not find the requested file path" in (guarded.answer or "")
    assert "No plausible replacement path" in guarded.uncertainties[0]


def test_fs_path_recovery_guard_ignores_echoed_invalid_output_path():
    from agents.fs_agent import (
        FsAgentResult,
        PathAnalysis,
        _apply_path_recovery_guard,
    )

    output = FsAgentResult(
        answer="Maybe this path.",
        summary="No result.",
        paths=["/skills/agentsystem.md"],
    )
    guarded = _apply_path_recovery_guard(
        output,
        PathAnalysis(invalid_paths=["/skills/agentsystem.md"]),
    )

    assert "could not find the requested file path" in (guarded.answer or "")
    assert "Please confirm the exact path" not in (guarded.answer or "")


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


def test_fs_usage_limit_error_is_not_reported_as_file_access_problem():
    from pydantic_ai.exceptions import UsageLimitExceeded

    from agents.fs_agent import _format_exception_report

    message = _format_exception_report(
        "read local files",
        UsageLimitExceeded("The next tool call(s) would exceed the tool_calls_limit."),
    )

    assert "tool-call budget" in message
    assert "file access problem" not in message


def test_fs_context_limit_error_is_not_reported_as_path_problem():
    from agents.fs_agent import _format_exception_report

    message = _format_exception_report(
        "summarize local papers",
        RuntimeError(
            "request (40196 tokens) exceeds the available context size "
            "(8192 tokens); exceed_context_size_error"
        ),
    )

    assert "exceeded its context limit" in message
    assert "not a file path or permission problem" in message
    assert "path needs to be corrected" not in message


def test_fs_small_pdf_is_always_selected_for_rag(monkeypatch, tmp_path):
    from agents import fs_agent

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "paper.pdf").write_bytes(b"%PDF-1.4\nsmall")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    assert fs_agent._paths_that_need_rag(["/docs/paper.pdf"]) == [
        "/docs/paper.pdf"
    ]


def test_fs_large_text_is_left_to_read_file_rag_branch(monkeypatch, tmp_path):
    from agents import fs_agent
    from tools.filesystem.types import DEFAULT_MAX_READ_CHARS

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "large.md").write_text(
        "A" * (DEFAULT_MAX_READ_CHARS + 1),
        encoding="utf-8",
    )
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)

    assert fs_agent._paths_that_need_rag(["/docs/large.md"]) == []


def test_fs_plan_worker_rag_uses_only_assigned_files(monkeypatch, tmp_path):
    from agents import fs_agent
    from agents.fs.contracts import PathAnalysis

    docs = tmp_path / "docs"
    paper_dir = docs / "papers" / "arxiv"
    paper_dir.mkdir(parents=True)
    for name in ("a.md", "b.md", "c.md"):
        (paper_dir / name).write_text(name, encoding="utf-8")
    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[Mount(host_path=docs, mount_point="/docs", mode="ro")]
        )
    )
    monkeypatch.setattr(fs_agent, "validator", validator)
    objective = "\n".join(
        [
            "Plan worker task:",
            "Original user prompt: Summarize all assigned local papers.",
            "Task objective: Process collection batch 1 of 2.",
            "Task kind: local_rag",
            "Query: Summarize every assigned local paper.",
            "Relevant local files:",
            "- /docs/papers/arxiv/a.md",
            "- /docs/papers/arxiv/b.md",
            "Return a concise result.",
        ]
    )
    analysis = PathAnalysis(
        resolved_paths=[
            "/docs/papers/arxiv",
            "/docs/papers/arxiv/a.md",
            "/docs/papers/arxiv/b.md",
        ]
    )

    assert fs_agent._preemptive_rag_paths(objective, analysis, ["/docs"]) == [
        "/docs/papers/arxiv/a.md",
        "/docs/papers/arxiv/b.md",
    ]
