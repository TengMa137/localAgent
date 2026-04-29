from agents.plan_agent import PlanOutput, _normalize_plan
from agents.query_policy import TaskKind, extract_urls
from agents.worker import TaskSpec


def test_query_policy_extracts_urls():
    assert extract_urls("Read https://example.com/a.") == ["https://example.com/a"]


def test_normalize_plan_preserves_planner_current_info_judgment():
    plan = PlanOutput(
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Check current pricing",
                requires_current_info=True,
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


def test_normalize_plan_converts_file_keyword_web_task_to_local_rag():
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

    assert normalized.tasks[0].kind == TaskKind.LOCAL_RAG
    assert "/skills/fitness/diet.md" in normalized.tasks[0].relevant_files
    assert "/skills/fitness/workout.md" in normalized.tasks[0].relevant_files
    assert "/skills/research/strategy.md" not in normalized.tasks[0].relevant_files
    assert "/skills/skill_editing.md" not in normalized.tasks[0].relevant_files


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
