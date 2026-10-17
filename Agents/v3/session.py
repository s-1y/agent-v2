"""按 session key 挂起 / 唤醒 OpenHands Conversation。

同一次运行的会话落在：
    <workspace>/.meta_agent/runs/<run_id>/conversations/<session_key>/<conversation_id.hex>/

OpenHands 会在 persistence_dir 下再拼一层 id.hex；外层目录用 session_key。
工作区根目录仍是项目本身，读写真的是那里的文件。另一次运行用另一个 run_id，
不会唤醒上一次的 n0、n1。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from openhands.sdk.conversation.persistence_const import BASE_STATE

# 稳定命名空间：同一 node_id 永远映射到同一个 ConversationID。
_NAMESPACE = uuid.UUID("a6e0f3c1-8b2d-4e9f-91a7-0c5d2b8e4f10")

META_AGENT_DIR = ".meta_agent"
CONVERSATIONS_DIR = "conversations"


def conversation_id_for(run_id: str, session_key: str) -> uuid.UUID:
    _require_run_id(run_id)
    return uuid.uuid5(_NAMESPACE, f"{run_id}\n{session_key}")


def conversations_root(workspace: Path, run_id: str) -> Path:
    _require_run_id(run_id)
    return Path(workspace) / META_AGENT_DIR / "runs" / run_id / CONVERSATIONS_DIR


def persistence_dir_for(workspace: Path, run_id: str, session_key: str) -> Path:
    """传给 Conversation 的 persistence_dir（其下还会有 id.hex 子目录）。"""
    path = conversations_root(workspace, run_id) / session_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_dir_for(workspace: Path, run_id: str, session_key: str) -> Path:
    """真正的 base_state.json / events/ 所在目录。"""
    return (
        persistence_dir_for(workspace, run_id, session_key)
        / conversation_id_for(run_id, session_key).hex
    )


def has_persisted_session(workspace: Path, run_id: str, session_key: str) -> bool:
    return (store_dir_for(workspace, run_id, session_key) / BASE_STATE).is_file()


def _require_run_id(run_id: str) -> None:
    if not str(run_id).strip():
        raise ValueError("run_id 不能为空，否则多次运行会共用同一批节点会话")
