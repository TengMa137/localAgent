"""Small local JSON API for frontend speech integration."""

from __future__ import annotations

import argparse
import asyncio
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

from run_agents import ChatSession, handle_turn
from speech.qwen3 import (
    Qwen3ASRConfig,
    Qwen3ASRProvider,
    Qwen3TTSConfig,
    Qwen3TTSProvider,
    TTSResult,
)


_sessions: dict[str, ChatSession] = {}


class VoiceAPIHandler(BaseHTTPRequestHandler):
    server_version = "LocalAgentVoiceAPI/0.1"

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        if self.path == "/speech/models":
            asr_config = Qwen3ASRConfig()
            tts_config = Qwen3TTSConfig()
            self._send_json(
                {
                    "asr": {
                        "provider": "qwen3-asr-llamacpp",
                        "model": asr_config.model,
                        "base_url": asr_config.base_url,
                    },
                    "tts": {
                        "provider": "qwen3-tts-cpp",
                        "model_dir": tts_config.model_dir,
                        "cli_path": tts_config.cli_path,
                    },
                }
            )
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/speech/asr":
            self._handle_asr(run_agent=False)
            return
        if self.path == "/speech/tts":
            self._handle_tts()
            return
        if self.path == "/agent/voice-turn":
            self._handle_asr(run_agent=True)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_asr(self, *, run_agent: bool) -> None:
        try:
            payload = self._read_json()
            audio_base64 = payload.get("audio_base64")
            if not isinstance(audio_base64, str) or not audio_base64.strip():
                self._send_error(HTTPStatus.BAD_REQUEST, "audio_base64 is required")
                return

            result = asyncio.run(
                Qwen3ASRProvider().transcribe_base64(
                    audio_base64,
                    mime_type=str(payload.get("mime_type") or "audio/wav"),
                    language=payload.get("language"),
                )
            )
            response: dict[str, Any] = {
                "text": result.text,
                "language": result.language,
                "provider": result.provider,
            }

            if run_agent:
                session_id = str(payload.get("session_id") or uuid4())
                session = _sessions.setdefault(session_id, ChatSession())
                reply = asyncio.run(
                    handle_turn(
                        result.text,
                        session,
                        debug=bool(payload.get("debug", False)),
                    )
                )
                response["session_id"] = session_id
                response["assistant_reply"] = reply
                if bool(payload.get("tts", False)):
                    speech = self._synthesize_text(reply, payload)
                    response["assistant_audio_base64"] = speech.audio_base64
                    response["assistant_audio_mime_type"] = speech.mime_type
                    response["tts_provider"] = speech.provider

            self._send_json(response)
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_tts(self) -> None:
        try:
            payload = self._read_json()
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send_error(HTTPStatus.BAD_REQUEST, "text is required")
                return

            result = self._synthesize_text(text, payload)

            self._send_json(
                {
                    "audio_base64": result.audio_base64,
                    "mime_type": result.mime_type,
                    "provider": result.provider,
                }
            )
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _synthesize_text(self, text: str, payload: dict[str, Any]) -> TTSResult:
        provider = Qwen3TTSProvider()
        reference_audio_base64 = payload.get("reference_audio_base64")
        if isinstance(reference_audio_base64, str) and reference_audio_base64.strip():
            return asyncio.run(
                provider.synthesize_with_reference_base64(
                    text,
                    reference_audio_base64,
                    reference_mime_type=str(
                        payload.get("reference_mime_type") or "audio/wav"
                    ),
                )
            )
        return asyncio.run(
            provider.synthesize(
                text,
                reference_audio_path=payload.get("reference_audio_path"),
            )
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if not body:
            return {}
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local agent voice API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VoiceAPIHandler)
    print(f"Voice API listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
