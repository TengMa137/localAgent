#!/usr/bin/env sh
set -eu

mkdir -p chat_history .memory

docker compose run --rm agent-cli
