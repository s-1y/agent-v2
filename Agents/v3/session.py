"""按 session key 挂起 / 唤醒 OpenHands Conversation。

同一次运行的会话落在：
    <workspace>/.meta_agent/runs/<run_id>/conversations/<session_key>/<conversation_id.hex>/

同一次运行的开工文档落在：
    <workspace>/.meta_agent/runs/<run_id>/docs/verN/

OpenHands 会在 persistence_dir 下再拼一层 id.hex；外层目录用 session_key。
工作区根目录仍是项目本身，读写真的是那里的文件。另一次运行用另一个 run_id，
不会唤醒上一次的 Root、Plan、Worker 0，也看不到上一次的开工文档。
同一次运行里的设计和开发看的是同一份文档。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from openhands.sdk.conversation.persistence_const import BASE_STATE

# 稳定命名空间：同一 node_id 永远映射到同一个 ConversationID。
_NAMESPACE = uuid.UUID("a6e0f3c1-8b2d-4e9f-91a7-0c5d2b8e4f10")

META_AGENT_DIR = ".meta_agent"
CONVERSATIONS_DIR = "conversations"
PLAN_DOCS_DIR = "docs"


def conversation_id_for(run_id: str, session_key: str) -> uuid.UUID:
    _require_run_id(run_id)
    return uuid.uuid5(_NAMESPACE, f"{run_id}\n{session_key}")


def run_root(workspace: Path, run_id: str) -> Path:
    """一次运行的根目录。对话和开工文档都放在这里。"""
    _require_run_id(run_id)
    return Path(workspace) / META_AGENT_DIR / "runs" / run_id


def conversations_root(workspace: Path, run_id: str) -> Path:
    return run_root(workspace, run_id) / CONVERSATIONS_DIR


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


def _plan_docs_dir(workspace: Path, run_id: str, version: int) -> Path:
    return run_root(workspace, run_id) / PLAN_DOCS_DIR / f"ver{version}"


def plan_doc_path(workspace: Path, run_id: str, version: int = 1) -> Path:
    """这一次运行的某一版内部计划。ver1 写完就留在原地，下一版只写 ver2、ver3。"""
    path = _plan_docs_dir(workspace, run_id, version)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plan_docs_ready(workspace: Path, run_id: str, version: int = 1) -> bool:
    """这一版文件夹里至少有一个非空文件，才算手册写好了。"""
    folder = _plan_docs_dir(workspace, run_id, version)
    if not folder.is_dir():
        return False
    for file in folder.rglob("*"):
        if file.is_file() and file.stat().st_size > 0:
            return True
    return False


def has_persisted_session(workspace: Path, run_id: str, session_key: str) -> bool:
    return (store_dir_for(workspace, run_id, session_key) / BASE_STATE).is_file()


def _require_run_id(run_id: str) -> None:
    if not str(run_id).strip():
        raise ValueError("run_id 不能为空，否则多次运行会共用同一批节点会话")
