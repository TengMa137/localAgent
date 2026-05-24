FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /workspace

COPY rag_lib ./rag_lib
COPY localAgent.worktrees/feature-auth ./localAgent.worktrees/feature-auth

WORKDIR /workspace/localAgent.worktrees/feature-auth
RUN uv sync --no-dev

ENV PYTHONPATH=/workspace/localAgent.worktrees/feature-auth/src \
    LOCALAGENT_STATE_DIR=/data/state \
    LOCALAGENT_DOCS_DIR=/data/docs \
    LOCALAGENT_SKILLS_DIR=/data/skills \
    LOCALAGENT_MCP_URL=http://mcp-server:8000/sse

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/state /data/docs /data/skills \
    && chown -R app:app /workspace /data

USER app

EXPOSE 8080
CMD [".venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
