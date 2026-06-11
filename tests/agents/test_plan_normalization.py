import pytest

from agents.plan_agent import (
    MAX_TASKS_PER_PLAN,
    PlanFileResolver,
    PlanOutput,
    PlannerInput,
    SessionState,
    _collection_findings_report,
    _format_plan_handoff,
    _run_research_loop,
    _normalize_plan,
)
from agents.runtime.query_policy import (
    TaskKind,
    ambiguously_references_local_artifact,
    explicitly_requests_local_source,
    explicitly_requests_web,
    extract_urls,
    infer_task_kind,
    likely_requires_current_info,
    requests_collection_plan,
    requests_file_operation,
    requests_local_discovery,
    requests_topic_file_discovery,
)
from agents.worker import TaskSpec
from agents.runtime.turn_context import EvidenceItem


def test_query_policy_extracts_urls():
    assert extract_urls("Read https://example.com/a.") == ["https://example.com/a"]


def test_query_policy_has_no_default_retrieval_kind():
    assert infer_task_kind("Explain recursion") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ok, now explain the architecture", False),
        ("describe the current implementation", False),
        ("read the current file", False),
        ("who is the current president?", True),
        ("what is the current package version?", True),
        ("show recent language model research", True),
        ("check tomorrow's weather", True),
        ("explain rate limiting", False),
        ("design a pricing page", False),
        ("implement live reload", False),
        ("schedule background jobs", False),
        ("score model predictions", False),
        ("check the exchange rate", True),
        ("gold price", True),
        ("who currently leads the agency?", True),
    ],
)
def test_query_policy_current_info_signals_are_conservative(text, expected):
    assert likely_requires_current_info(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("search the web for agent patterns", True),
        ("look up online examples", True),
        ("verify on the internet", True),
        ("download the paper", True),
        ("run a web app locally", False),
        ("search local files", False),
        ("check agentsystem.md", False),
    ],
)
def test_query_policy_explicit_web_phrases_name_the_source(text, expected):
    assert explicitly_requests_web(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("check agentsystem.md", True),
        ("summarize AGENT_SYSTEM.md", True),
        ("based on config.yaml, explain the setup", True),
        ("mention agentsystem.md in the answer", False),
    ],
)
def test_query_policy_local_file_actions_require_an_operation(text, expected):
    assert requests_file_operation(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("summarize the paper", True),
        ("check my recent notes", True),
        ("inspect the saved document", True),
        ("find it in the same folder", True),
        ("what about 2605.00080v1 in the same folder?", True),
        ("find the latest paper", True),
        ("search the web for the paper", False),
        ("explain how files work in Python", False),
    ],
)
def test_query_policy_local_discovery_is_narrow_and_local_first(text, expected):
    assert requests_local_discovery(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("summarize the authentication documentation", True),
        ("find files related to world models", True),
        ("review architecture notes", True),
        ("edit the authentication documentation", False),
        ("write a new local note", False),
        ("explain authentication", False),
    ],
)
def test_query_policy_topic_file_discovery_is_read_only(text, expected):
    assert requests_topic_file_discovery(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("now summarize all papers locally in parallel", True),
        ("I mean all papers under docs/papers/arxiv", True),
        (
            "you should first check all papers locally then read and summarize "
            "them in parallel",
            True,
        ),
        ("summarize one local paper", False),
        ("list all files under docs", False),
        ("find papers about world models", False),
    ],
)
def test_query_policy_collection_plan_is_narrow(text, expected):
    assert requests_collection_plan(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "check local papers regarding world models",
        "search local files for agent routing",
        "review my architecture notes",
        "inspect the downloaded document",
    ],
)
def test_query_policy_explicit_local_source_overrides_webish_words(text):
    assert explicitly_requests_local_source(text)
    assert requests_local_discovery(text)
    assert infer_task_kind(text) == TaskKind.LOCAL_RAG


def test_query_policy_referential_artifact_is_ambiguous_not_explicit_local():
    text = "summarize the paper and explain its architecture"

    assert ambiguously_references_local_artifact(text)
    assert not explicitly_requests_local_source(text)


def test_query_policy_now_does_not_force_web_but_explicit_web_does():
    assert infer_task_kind("ok, now explain this design") is None
    assert infer_task_kind("search the web for this design") == TaskKind.WEB_SEARCH


def test_query_policy_external_arxiv_lookup_uses_web_search():
    assert (
        infer_task_kind("search the web for arXiv 2605.00080")
        == TaskKind.WEB_SEARCH
    )


def test_query_policy_resolved_local_files_win_over_external_identifiers():
    assert (
        infer_task_kind(
            "summarize arXiv 2605.00080",
            matched_files=["/docs/papers/2605.00080.md"],
        )
        == TaskKind.LOCAL_RAG
    )


def test_query_policy_local_reference_wins_over_incidental_recent_word():
    assert infer_task_kind("check my recent notes") == TaskKind.LOCAL_RAG
    assert infer_task_kind("summarize the paper") == TaskKind.LOCAL_RAG
    assert infer_task_kind("find the latest paper") == TaskKind.LOCAL_RAG


def test_query_policy_explicit_url_wins_over_referential_artifact():
    assert (
        infer_task_kind("summarize the paper at https://example.com/paper")
        == TaskKind.URL_CRAWL
    )


def test_plan_file_resolver_expands_explicit_collection_directory():
    resolver = PlanFileResolver(
        [
            "/docs/papers/arxiv/a.md",
            "/docs/papers/arxiv/b.pdf",
            "/docs/papers/arxiv/.DS_Store",
            "/docs/papers/other.md",
            "/docs/notes.txt",
        ]
    )

    assert resolver.resolve_collection(
        "summarize all papers under docs/papers/arxiv",
        matched_files=[],
    ) == [
        "/docs/papers/arxiv/a.md",
        "/docs/papers/arxiv/b.pdf",
    ]


def test_plan_file_resolver_ignores_hidden_non_documents_in_paper_collection():
    resolver = PlanFileResolver(
        [
            "/docs/papers/.DS_Store",
            "/docs/papers/arxiv/paper.md",
            "/docs/papers/arxiv/cache.bin",
        ]
    )

    assert resolver.resolve_collection(
        "summarize all papers locally in parallel",
        matched_files=[],
    ) == ["/docs/papers/arxiv/paper.md"]


def test_normalize_plan_batches_every_collection_file():
    files = [
        "/docs/papers/arxiv/a.md",
        "/docs/papers/arxiv/b.md",
        "/docs/papers/arxiv/c.pdf",
        "/docs/papers/arxiv/d.md",
    ]
    resolver = PlanFileResolver(files)
    from agents.plan_agent import PlanNormalizer

    normalized = PlanNormalizer(
        objective="summarize all papers locally in parallel",
        matched_files=[],
        as_of="Sunday, 7 June 2026, 10:00 UTC",
        resolver=resolver,
        max_tasks=3,
    ).normalize(PlanOutput(initial_answer="There is only one paper."))

    assert len(normalized.tasks) == 3
    assert {
        path
        for task in normalized.tasks
        for path in task.relevant_files
    } == set(files)
    assert all(task.kind == TaskKind.LOCAL_RAG for task in normalized.tasks)
    assert all("Do not omit assigned artifacts" in task.objective for task in normalized.tasks)
    assert normalized.initial_answer is None


def test_normalize_plan_groups_same_stem_markdown_and_pdf_as_one_paper():
    files = [
        "/docs/papers/arxiv/a.md",
        "/docs/papers/arxiv/a.pdf",
        "/docs/papers/arxiv/b.md",
    ]
    resolver = PlanFileResolver(files)
    from agents.plan_agent import PlanNormalizer

    normalized = PlanNormalizer(
        objective="summarize all papers under docs/papers/arxiv in parallel",
        matched_files=[],
        as_of="Sunday, 7 June 2026, 10:00 UTC",
        resolver=resolver,
        max_tasks=3,
    ).normalize(PlanOutput(tasks=[]))

    assert len(normalized.tasks) == 2
    assert normalized.tasks[0].relevant_files == files[:2]
    assert normalized.tasks[1].relevant_files == files[2:]


def test_collection_findings_report_forwards_grounded_worker_answers():
    state = SessionState(
        user_query="summarize all papers",
        evidence_items=[
            EvidenceItem(
                task_id="1",
                objective="batch one",
                agent="fs_agent",
                answer="Paper A: grounded summary.",
                useful=True,
                sources=["/docs/papers/a.md"],
            ),
            EvidenceItem(
                task_id="2",
                objective="batch two",
                agent="fs_agent",
                answer="Paper B: grounded summary.",
                useful=True,
                sources=["/docs/papers/b.pdf"],
            ),
        ],
        uncertainties=["Paper B had no abstract metadata."],
    )

    report = _collection_findings_report(state)

    assert "Paper A: grounded summary." in report
    assert "Paper B: grounded summary." in report
    assert "Paper B had no abstract metadata." in report
    assert "internal reasoning" not in report


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
        objective=("check out /skills/fitness/diet.md and /skills/fitness/workout.md"),
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


def test_normalize_plan_drops_initial_answer_for_current_info_objective():
    normalized = _normalize_plan(
        PlanOutput(initial_answer="Ungrounded direct answer."),
        objective="What is the latest OpenAI API pricing?",
        matched_files=[],
        as_of="now",
    )

    assert normalized.initial_answer is None
    assert normalized.tasks[0].kind == TaskKind.WEB_SEARCH
    assert normalized.tasks[0].requires_current_info is True


def test_normalize_plan_uses_web_search_for_unresolved_arxiv_id():
    normalized = _normalize_plan(
        PlanOutput(initial_answer="Ungrounded paper summary."),
        objective="Summarize arXiv 2605.00080",
        matched_files=[],
        as_of="now",
    )

    assert normalized.initial_answer is None
    assert normalized.tasks[0].kind == TaskKind.WEB_SEARCH
    assert normalized.tasks[0].query == "Summarize arXiv 2605.00080"


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


def test_plan_handoff_keeps_compact_task_ledger():
    state = SessionState(user_query="Compare local and current docs")
    state.completed_tasks = ["Read local docs"]
    state.findings = ["Detailed specialist result that should stay out of notes"]
    state.uncertainties = ["Need current API confirmation"]
    state.sources = ["/docs/local.md"]

    handoff = _format_plan_handoff(
        answer="Use the new API shape.",
        state=state,
        planned_tasks=[
            TaskSpec(
                kind=TaskKind.LOCAL_RAG,
                objective="Read local docs",
                relevant_files=["/docs/local.md"],
            ),
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective="Check current API docs",
                query="current API docs",
            ),
        ],
        as_of="now",
        time_sensitive=True,
    )

    assert "Forwardable answer:\nUse the new API shape." in handoff
    assert "Orchestrator notes:" in handoff
    assert "Tasks planned: 2; completed: 1" in handoff
    assert "Pending tasks: Check current API docs" in handoff
    assert "Findings available: 1" in handoff
    assert "Skipped unhelpful results: none" in handoff
    assert (
        "Detailed specialist result" not in handoff.split("Orchestrator notes:", 1)[1]
    )


@pytest.mark.asyncio
async def test_research_loop_runs_planned_batches_without_reflection(monkeypatch):
    from agents import plan_agent

    worker_batches: list[list[str]] = []

    async def fake_workers(tasks):
        worker_batches.append([task.objective for task in tasks])
        return [
            {
                "status": "ok",
                "key_findings": [f"Evidence for {task.objective}"],
                "uncertainties": [],
                "suggested_next_steps": [],
                "cited_node_ids": [],
            }
            for task in tasks
        ]

    monkeypatch.setattr(plan_agent, "_run_workers_limited", fake_workers)

    state = SessionState(user_query="answer carefully")
    used_current_info = await _run_research_loop(
        objective="answer carefully",
        matched_files=[],
        as_of="now",
        state=state,
        tasks=[
            TaskSpec(kind=TaskKind.WEB_SEARCH, objective=f"Planned task {idx}")
            for idx in range(5)
        ],
    )

    assert worker_batches == [
        ["Planned task 0", "Planned task 1", "Planned task 2"],
        ["Planned task 3", "Planned task 4"],
    ]
    assert state.findings == [
        "Evidence for Planned task 0",
        "Evidence for Planned task 1",
        "Evidence for Planned task 2",
        "Evidence for Planned task 3",
        "Evidence for Planned task 4",
    ]
    assert used_current_info is False


@pytest.mark.asyncio
async def test_research_loop_records_pending_tasks_after_iteration_budget(monkeypatch):
    from agents import plan_agent

    async def fake_workers(tasks):
        return [
            {
                "status": "ok",
                "key_findings": [f"Evidence for {task.objective}"],
                "uncertainties": [],
                "suggested_next_steps": [],
                "cited_node_ids": [],
            }
            for task in tasks
        ]

    monkeypatch.setattr(plan_agent, "_run_workers_limited", fake_workers)

    state = SessionState(user_query="answer carefully")
    used_current_info = await _run_research_loop(
        objective="answer carefully",
        matched_files=[],
        as_of="now",
        state=state,
        tasks=[
            TaskSpec(
                kind=TaskKind.WEB_SEARCH,
                objective=f"Planned task {idx}",
                requires_current_info=idx == 0,
            )
            for idx in range(5)
        ],
        max_iterations=1,
    )

    assert used_current_info is True
    assert state.completed_tasks == [
        "Planned task 0",
        "Planned task 1",
        "Planned task 2",
    ]
    assert any(
        "iteration budget" in uncertainty
        and "Planned task 3" in uncertainty
        and "Planned task 4" in uncertainty
        for uncertainty in state.uncertainties
    )


@pytest.mark.asyncio
async def test_research_loop_skips_unhelpful_worker_results(monkeypatch):
    from agents import plan_agent

    async def fake_workers(tasks):
        return [
            {
                "task_id": "useful",
                "status": "done",
                "agent": "web_agent",
                "answer": "The docs now recommend v2.",
                "useful": True,
                "key_findings": ["The docs now recommend v2."],
                "uncertainties": ["Version rollout is ongoing."],
                "suggested_next_steps": [],
                "cited_node_ids": ["https://example.com/docs"],
            },
            {
                "task_id": "missing",
                "status": "done",
                "agent": "fs_agent",
                "answer": None,
                "useful": False,
                "key_findings": [],
                "uncertainties": ["Requested path not found: /docs/missing.md."],
                "suggested_next_steps": [],
                "cited_node_ids": [],
            },
        ]

    monkeypatch.setattr(plan_agent, "_run_workers_limited", fake_workers)

    state = SessionState(user_query="compare useful and missing evidence")
    await _run_research_loop(
        objective="compare useful and missing evidence",
        matched_files=[],
        as_of="now",
        state=state,
        tasks=[
            TaskSpec(kind=TaskKind.WEB_SEARCH, objective="Read current docs"),
            TaskSpec(kind=TaskKind.LOCAL_RAG, objective="Read missing local file"),
        ],
    )

    assert state.findings == ["The docs now recommend v2."]
    assert state.sources == ["https://example.com/docs"]
    assert state.uncertainties == ["Version rollout is ongoing."]
    assert state.skipped_tasks == ["Read missing local file"]
    assert state.evidence_items[0].agent == "web_agent"
