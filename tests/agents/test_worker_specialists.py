import pytest

from agents.runtime.query_policy import TaskKind
from agents.worker import SYNTHESIS_SYSTEM_PROMPT, TaskSpec, _run_worker


def test_synthesis_prompt_preserves_filesystem_candidate_confirmation():
    assert "give a short heads-up" in SYNTHESIS_SYSTEM_PROMPT
    assert "exact-path confirmation" in SYNTHESIS_SYSTEM_PROMPT
    assert "file-not-found" in SYNTHESIS_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_local_worker_delegates_to_fs_agent(monkeypatch):
    from agents import worker

    calls: list[str] = []

    async def fake_run_fs_task(objective: str) -> str:
        calls.append(objective)
        return "fs result"

    async def fake_run_web_task(_objective: str) -> str:
        raise AssertionError("local tasks must not call web_agent")

    monkeypatch.setattr(worker, "run_fs_task", fake_run_fs_task)
    monkeypatch.setattr(worker, "run_web_task", fake_run_web_task)

    result = await _run_worker(
        TaskSpec(
            kind=TaskKind.LOCAL_FILES,
            objective="Read the config",
            query="config setting",
            relevant_files=["/repo/config.md"],
            user_prompt="What is in the config?",
        )
    )

    assert result["status"] == "done"
    assert result["key_findings"] == ["fs result"]
    assert result["answer"] == "fs result"
    assert result["useful"] is True
    assert result["cited_node_ids"] == ["/repo/config.md"]
    assert "Task kind: local_files" in calls[0]
    assert "/repo/config.md" in calls[0]


@pytest.mark.asyncio
async def test_web_worker_delegates_to_web_agent(monkeypatch):
    from agents import worker

    calls: list[str] = []

    async def fake_run_fs_task(_objective: str) -> str:
        raise AssertionError("web tasks must not call fs_agent")

    async def fake_run_web_task(objective: str) -> str:
        calls.append(objective)
        return "web result"

    monkeypatch.setattr(worker, "run_fs_task", fake_run_fs_task)
    monkeypatch.setattr(worker, "run_web_task", fake_run_web_task)

    result = await _run_worker(
        TaskSpec(
            kind=TaskKind.URL_CRAWL,
            objective="Summarize the docs",
            query="docs summary",
            urls=["https://example.com/docs"],
            user_prompt="Read https://example.com/docs",
        )
    )

    assert result["status"] == "done"
    assert result["key_findings"] == ["web result"]
    assert result["answer"] == "web result"
    assert result["useful"] is True
    assert result["cited_node_ids"] == ["https://example.com/docs"]
    assert "Task kind: url_crawl" in calls[0]
    assert "https://example.com/docs" in calls[0]


@pytest.mark.asyncio
async def test_scholarly_web_worker_delegates_to_web_agent(monkeypatch):
    from agents import worker

    calls: list[str] = []

    async def fake_run_fs_task(_objective: str) -> str:
        raise AssertionError("web tasks must not call fs_agent")

    async def fake_run_web_task(objective: str) -> str:
        calls.append(objective)
        return "paper result"

    monkeypatch.setattr(worker, "run_fs_task", fake_run_fs_task)
    monkeypatch.setattr(worker, "run_web_task", fake_run_web_task)

    result = await _run_worker(
        TaskSpec(
            kind=TaskKind.WEB_SEARCH,
            objective="Read arXiv 2401.12345",
            query="2401.12345",
        )
    )

    assert result["status"] == "done"
    assert result["key_findings"] == ["paper result"]
    assert result["answer"] == "paper result"
    assert result["useful"] is True
    assert "Task kind: web_search" in calls[0]


@pytest.mark.asyncio
async def test_worker_propagates_specialist_uncertainties(monkeypatch):
    from agents import worker

    async def fake_run_web_task(_objective: str) -> str:
        return (
            "Forwardable answer:\n"
            "No answer returned.\n\n"
            "Orchestrator notes:\n"
            "- Summary: Search did not find crawlable sources.\n"
            "- Uncertainties: No URLs were selected or crawled.; Try a narrower query."
        )

    monkeypatch.setattr(worker, "run_web_task", fake_run_web_task)

    result = await _run_worker(
        TaskSpec(
            kind=TaskKind.WEB_SEARCH,
            objective="Find current docs",
            query="current docs",
            requires_current_info=True,
        )
    )

    assert result["status"] == "done"
    assert result["answer"] is None
    assert result["useful"] is False
    assert result["key_findings"] == []
    assert result["uncertainties"] == [
        "No URLs were selected or crawled.",
        "Try a narrower query.",
    ]


@pytest.mark.asyncio
async def test_worker_drops_file_not_found_from_synthesis_findings(monkeypatch):
    from agents import worker

    async def fake_run_fs_task(_objective: str) -> str:
        return (
            "Forwardable answer:\n"
            "I could not find the requested file path (/docs/missing.md) under the readable roots: /docs.\n\n"
            "Orchestrator notes:\n"
            "- Summary: File not found.\n"
            "- Uncertainties: Invalid path hint(s): /docs/missing.md."
        )

    monkeypatch.setattr(worker, "run_fs_task", fake_run_fs_task)

    result = await _run_worker(
        TaskSpec(
            kind=TaskKind.LOCAL_FILES,
            objective="Read missing file",
            relevant_files=["/docs/missing.md"],
        )
    )

    assert result["status"] == "done"
    assert result["useful"] is False
    assert result["answer"] is None
    assert result["key_findings"] == []
