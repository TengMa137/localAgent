import pytest

from agents.plan_agent import (
    MAX_TASKS_PER_PLAN,
    PlanFileResolver,
    PlanOutput,
    PlannerInput,
    SessionState,
    _run_research_loop,
    _normalize_plan,
)
from agents.runtime.query_policy import TaskKind, extract_urls, infer_task_kind
from agents.worker import ReflectionOutput, TaskSpec


def test_query_policy_extracts_urls():
    assert extract_urls("Read https://example.com/a.") == ["https://example.com/a"]


def test_query_policy_has_no_default_retrieval_kind():
    assert infer_task_kind("Explain recursion") is None


def test_normalize_plan_forces_web_search_current_info():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Check current pricing",
                requires_current_info=False,
            )
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective="What is the latest OpenAI API pricing?",
        matched_files=[],
        as_of="Wednesday, 29 April 2026, 10:00 UTC",
    )

    assert normalized.tasks[0].kind == TaskKind.WEB_SEARCH
    assert normalized.tasks[0].requires_current_info is True
    assert normalized.tasks[0].as_of == "Wednesday, 29 April 2026, 10:00 UTC"


def test_normalize_plan_filters_unresolved_files_and_adds_local_task():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.LOCAL_RAG,
                objective="Read files",
                relevant_files=["/docs/ok.md", "/docs/missing.md"],
            )
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective="Summarize my file",
        matched_files=["/docs/ok.md"],
        as_of="now",
    )

    assert normalized.tasks[0].kind == TaskKind.LOCAL_RAG
    assert normalized.tasks[0].relevant_files == ["/docs/ok.md"]
    assert normalized.tasks[0].user_prompt == "Summarize my file"


def test_normalize_plan_repairs_relative_path_against_readable_files():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.LOCAL_RAG,
                objective="Read diet skill",
                relevant_files=["/fitness/diet.md"],
            )
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective="check out fitness skills",
        matched_files=[],
        as_of="now",
    )

    assert "/skills/fitness/diet.md" in normalized.tasks[0].relevant_files


def test_plan_file_resolver_does_not_fuzzy_match_query_terms():
    resolver = PlanFileResolver(
        ["/skills/fitness/diet.md", "/skills/fitness/workout.md"]
    )

    assert resolver.resolve("check out fitness skills", matched_files=[]) == []


def test_normalize_plan_does_not_convert_file_keyword_web_task_to_local_rag():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Analyze the fitness skills system",
                query="fitness skills game",
            )
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective="check out fitness skills",
        matched_files=[],
        as_of="now",
    )

    assert normalized.tasks[0].kind == TaskKind.WEB_SEARCH
    assert normalized.tasks[0].relevant_files == []


def test_normalize_plan_adds_required_local_task_for_objective_files():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Analyze implementation details",
                query="implementation details",
            )
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective=(
            "check out /skills/fitness/diet.md and "
            "/skills/fitness/workout.md"
        ),
        matched_files=[],
        as_of="now",
    )

    assert normalized.tasks[0].kind == TaskKind.LOCAL_RAG
    assert "/skills/fitness/diet.md" in normalized.tasks[0].relevant_files
    assert "/skills/fitness/workout.md" in normalized.tasks[0].relevant_files
    assert normalized.tasks[1].kind == TaskKind.WEB_SEARCH


def test_normalize_plan_adds_required_local_task_for_matched_files():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Analyze implementation details",
                query="implementation details",
            )
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective="analyze the known implementation",
        matched_files=["/skills/fitness/diet.md"],
        as_of="now",
    )

    assert normalized.tasks[0].kind == TaskKind.LOCAL_RAG
    assert normalized.tasks[0].relevant_files == ["/skills/fitness/diet.md"]
    assert normalized.tasks[1].kind == TaskKind.WEB_SEARCH


def test_normalize_plan_extracts_file_path_from_task_objective():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Summarize the fitness/diet.md file",
                query="fitness/diet.md",
            )
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective="check out fitness skills",
        matched_files=[],
        as_of="now",
    )

    assert normalized.tasks[0].kind == TaskKind.LOCAL_RAG
    assert normalized.tasks[0].relevant_files == ["/skills/fitness/diet.md"]


def test_normalize_plan_infers_url_kind_when_planner_omits_kind():
    normalized = _normalize_plan(
        PlanOutput(
            tasks=[
                TaskSpec(
                    objective="Summarize https://example.com/docs",
                    query="https://example.com/docs",
                )
            ]
        ),
        objective="Summarize https://example.com/docs",
        matched_files=[],
        as_of="now",
    )

    assert normalized.tasks[0].kind == TaskKind.URL_CRAWL
    assert normalized.tasks[0].urls == ["https://example.com/docs"]


def test_normalize_plan_drops_untyped_task_without_structural_route():
    normalized = _normalize_plan(
        PlanOutput(tasks=[TaskSpec(objective="Explain recursion")]),
        objective="Explain recursion",
        matched_files=[],
        as_of="now",
    )

    assert normalized.tasks == []


def test_normalize_plan_keeps_initial_answer_with_file_context():
    plan = PlanOutput(initial_answer="The preview fully answers this.")

    normalized = _normalize_plan(
        plan,
        objective="summarize the known file",
        matched_files=["/skills/fitness/diet.md"],
        as_of="now",
    )

    assert normalized.initial_answer == "The preview fully answers this."
    assert normalized.tasks == []


def test_normalize_plan_drops_initial_answer_for_url_objective():
    normalized = _normalize_plan(
        PlanOutput(initial_answer="Ungrounded direct answer."),
        objective="Summarize https://example.com/docs",
        matched_files=[],
        as_of="now",
    )

    assert normalized.initial_answer is None
    assert normalized.tasks


def test_normalize_plan_trusts_initial_answer_without_structural_web_signal():
    normalized = _normalize_plan(
        PlanOutput(initial_answer="Ungrounded direct answer."),
        objective="What is the latest OpenAI API pricing?",
        matched_files=[],
        as_of="now",
    )

    assert normalized.initial_answer == "Ungrounded direct answer."
    assert normalized.tasks == []


def test_normalize_plan_keeps_initial_answer_for_non_web_terminal_issue():
    normalized = _normalize_plan(
        PlanOutput(initial_answer="I could not resolve the requested path."),
        objective="read missing local file",
        matched_files=[],
        as_of="now",
    )

    assert normalized.initial_answer == "I could not resolve the requested path."
    assert normalized.tasks == []


def test_empty_plan_does_not_invent_default_web_task():
    normalized = _normalize_plan(
        PlanOutput(),
        objective="What changed in the package API?",
        matched_files=[],
        as_of="now",
    )

    assert normalized.initial_answer is None
    assert normalized.tasks == []


def test_required_url_task_is_not_dropped_when_plan_is_full():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective=f"Search topic {idx}",
                query=f"topic {idx}",
            )
            for idx in range(MAX_TASKS_PER_PLAN)
        ]
    )

    normalized = _normalize_plan(
        plan,
        objective="compare the plan with https://example.com/docs",
        matched_files=[],
        as_of="now",
    )

    assert normalized.tasks[0].kind == TaskKind.URL_CRAWL
    assert normalized.tasks[0].urls == ["https://example.com/docs"]
    assert len(normalized.tasks) == MAX_TASKS_PER_PLAN


def test_planner_input_renders_known_file_context():
    planner_input = PlannerInput(
        objective="compare local code with current docs",
        matched_files=[],
        file_paths=["/docs/code.md"],
        file_context="PATH: /docs/code.md\nPREVIEW:\ncode",
        resolver=PlanFileResolver([]),
    )

    prompt = planner_input.render_prompt()

    assert "Resolved file paths" in prompt
    assert "/docs/code.md" in prompt
    assert "PATH: /docs/code.md" in prompt


@pytest.mark.asyncio
async def test_research_loop_runs_followups_when_reflection_confidence_is_low(
    monkeypatch,
):
    from agents import plan_agent

    worker_batches: list[list[str]] = []

    async def fake_workers(tasks):
        worker_batches.append([task.objective for task in tasks])
        if len(worker_batches) == 1:
            return [
                {
                    "status": "ok",
                    "key_findings": [],
                    "uncertainties": ["Need more evidence."],
                    "suggested_next_steps": [],
                    "cited_node_ids": [],
                }
            ]
        return [
            {
                "status": "ok",
                "key_findings": ["Follow-up evidence."],
                "uncertainties": [],
                "suggested_next_steps": [],
                "cited_node_ids": [],
            }
        ]

    async def fake_reflect(**_kwargs):
        return ReflectionOutput(
            objective_complete=False,
            confidence="low",
            next_tasks=[
                TaskSpec(
                    kind=TaskKind.WEB_SEARCH,
                    objective="Run a targeted follow-up search",
                    query="targeted follow-up",
                )
            ],
        )

    monkeypatch.setattr(plan_agent, "_run_workers_limited", fake_workers)
    monkeypatch.setattr(plan_agent, "run_reflect_worker", fake_reflect)

    state = SessionState(user_query="answer carefully")
    used_current_info = await _run_research_loop(
        objective="answer carefully",
        matched_files=[],
        as_of="now",
        state=state,
        tasks=[TaskSpec(kind=TaskKind.WEB_SEARCH, objective="Initial search")],
    )

    assert worker_batches == [["Initial search"], ["Run a targeted follow-up search"]]
    assert state.findings == ["Follow-up evidence."]
    assert used_current_info is True


@pytest.mark.asyncio
async def test_research_loop_records_low_confidence_when_no_followup(monkeypatch):
    from agents import plan_agent

    async def fake_workers(_tasks):
        return [
            {
                "status": "ok",
                "key_findings": [],
                "uncertainties": ["Important gap remains."],
                "suggested_next_steps": [],
                "cited_node_ids": [],
            }
        ]

    async def fake_reflect(**_kwargs):
        return ReflectionOutput(
            objective_complete=False,
            confidence="low",
            next_tasks=[],
        )

    monkeypatch.setattr(plan_agent, "_run_workers_limited", fake_workers)
    monkeypatch.setattr(plan_agent, "run_reflect_worker", fake_reflect)

    state = SessionState(user_query="answer carefully")
    used_current_info = await _run_research_loop(
        objective="answer carefully",
        matched_files=[],
        as_of="now",
        state=state,
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Initial search",
                requires_current_info=True,
            )
        ],
    )

    assert used_current_info is True
    assert any("Reflection confidence was low" in note for note in state.uncertainties)
    assert any("returned no follow-up tasks" in note for note in state.uncertainties)


def test_reflection_output_coerces_legacy_numeric_confidence():
    assert ReflectionOutput(
        objective_complete=False,
        confidence=0.2,
        next_tasks=[],
    ).confidence == "low"
    assert ReflectionOutput(
        objective_complete=True,
        confidence=0.9,
        next_tasks=[],
    ).confidence == "high"
