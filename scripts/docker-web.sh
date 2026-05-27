#!/usr/bin/env sh
set -eu

env_file_value() {
  key="$1"
  fallback="$2"

  if [ ! -f .env ]; then
    printf '%s' "$fallback"
    return
  fi

  value="$(
    awk -F= -v key="$key" '
      $1 == key {
        value = substr($0, index($0, "=") + 1)
      }
      END {
        if (value != "") {
          print value
        }
      }
    ' .env
  )"

  if [ -n "$value" ]; then
    value="${value%%#*}"
    value="$(printf '%s' "$value" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/^"//; s/"$//; s/^'\''//; s/'\''$//')"
    printf '%s' "$value"
  else
    printf '%s' "$fallback"
  fi
}

bind_host="${LOCALAGENT_BIND:-$(env_file_value LOCALAGENT_BIND 127.0.0.1)}"
port="${LOCALAGENT_PORT:-$(env_file_value LOCALAGENT_PORT 8088)}"

if [ "$bind_host" = "0.0.0.0" ]; then
  open_host="127.0.0.1"
else
  open_host="$bind_host"
fi

url="http://${open_host}:${port}"

docker compose up --build -d agent-app

printf 'Waiting for %s/health' "$url"
attempt=0
until curl -fsS "$url/health" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    printf '\nTimed out waiting for %s/health\n' "$url" >&2
    docker compose ps >&2
    exit 1
  fi
  printf '.'
  sleep 1
done
printf '\nOpening %s\n' "$url"

if command -v open >/dev/null 2>&1; then
  open "$url"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$url" >/dev/null 2>&1 &
elif command -v start >/dev/null 2>&1; then
  start "$url"
else
  printf 'No browser opener found. Open %s manually.\n' "$url"
fi

docker compose ps
