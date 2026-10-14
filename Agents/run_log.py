"""Quiet console + full timeline files under logging/."""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from Agents.paths import LOGGING_DIR

_SKIP_EVENT_TYPES = {
    "ConversationStateUpdateEvent",
    "SystemPromptEvent",
    "TokenEvent",
    "StreamingDeltaEvent",
    "LLMCompletionLogEvent",
}

_THINK_TOOLS = {"think", "ThinkTool"}


def quiet_third_party_logs() -> None:
    """Mute LiteLLM / OpenHands / HTTP chatter on stderr."""
    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    warnings.filterwarnings(
        "ignore",
        message=r"Cost calculation failed:.*",
        category=UserWarning,
    )
    for name in (
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
        "httpx",
        "httpcore",
        "openai",
        "openhands",
        "openhands.sdk",
        "openhands.sdk.llm",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _event_type(event: object) -> str:
    return type(event).__name__


def _tool_name(event: object) -> str | None:
    name = getattr(event, "tool_name", None)
    return str(name) if name else None


def _event_dump(event: object) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        try:
            dumped = event.model_dump(mode="json")
            if isinstance(dumped, dict):
                dumped.setdefault("event_type", _event_type(event))
                return dumped
        except Exception:
            pass
    return {"event_type": _event_type(event), "repr": repr(event)}


def _console_line(event: object) -> str | None:
    kind = _event_type(event)
    if kind in _SKIP_EVENT_TYPES:
        return None
    tool = _tool_name(event)
    if kind == "ActionEvent":
        if tool in _THINK_TOOLS:
            return "思考"
        return f"工具 {tool}" if tool else "工具"
    if kind == "ObservationEvent":
        return None
    if kind == "AgentErrorEvent":
        return "工具出错"
    if kind == "MessageEvent":
        source = getattr(event, "source", "")
        if source == "agent":
            return "思考"
        return None
    if kind == "PauseEvent":
        return "暂停"
    return None


class RunLogger:
    """Write every event to logging/<id>.jsonl; print one short line to stdout."""

    def __init__(self, *, label: str, spec: dict[str, Any] | None = None):
        LOGGING_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{label}"
        self.jsonl_path = LOGGING_DIR / f"{self.run_id}.jsonl"
        self.text_path = LOGGING_DIR / f"{self.run_id}.txt"
        header = {
            "event_type": "run_start",
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "spec": spec or {},
        }
        self._append_json(header)
        self._append_text(f"run {self.run_id}")

    def on_event(self, event: object) -> None:
        payload = _event_dump(event)
        payload["timestamp"] = payload.get("timestamp") or datetime.now().isoformat()
        self._append_json(payload)
        kind = payload.get("event_type", _event_type(event))
        tool = _tool_name(event)
        extra = f" {tool}" if tool else ""
        self._append_text(f"{payload['timestamp']}  {kind}{extra}")
        line = _console_line(event)
        if line:
            print(line, flush=True)

    def note(self, message: str, **extra: Any) -> None:
        payload = {
            "event_type": "note",
            "timestamp": datetime.now().isoformat(),
            "message": message,
            **extra,
        }
        self._append_json(payload)
        self._append_text(f"{payload['timestamp']}  note  {message}")
        print(message, flush=True)

    def _append_json(self, payload: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _append_text(self, line: str) -> None:
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
