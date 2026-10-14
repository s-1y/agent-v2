"""跑一个节点的一个回合：起会话、装工具、数步、拿回决定。

同一次运行里，每个 node_id 对应一份可挂起的 OpenHands 会话：
    <workspace>/.meta_agent/runs/<run_id>/conversations/<node_id>:plan|self/...
回合结束 close() 释放工具；磁盘保留。同一次运行里再次开工按同一 node_id resume，
只追加一条继续消息，不再冷启动整页任务卡。另一次运行用另一个 run_id，从空会话开始。
发给 OpenHands 的 workspace 仍是项目根目录，读写的是那里的文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Agents.engine import (
    ActionEvent,
    Agent,
    AgentContext,
    Conversation,
    MessageEvent,
    TextContent,
    coding_tools,
    make_condenser,
    make_llm,
    readonly_tools,
)
from Agents.v2.context import (
    plan_system,
    self_work_system,
    turn_message,
)
from Agents.v2.session import (
    conversation_id_for,
    has_persisted_session,
    persistence_dir_for,
)
from Agents.v2.tools import (
    Decision,
    plan_decision_tools,
    self_work_decision_tools,
    take_decision,
)
from Agents.v2.tree import Node, Tree

_UNBOUNDED_ITERATION_BUDGET = 1_000_000
_SELF_SESSION_PHASES = {"3A_self_work", "3A2_summarize"}


@dataclass
class TurnResult:
    decision: Optional[Decision]
    steps: int
    note: str = ""
    text: str = ""
    resumed: bool = False
    error: bool = False


def _session_key(node: Node) -> str:
    suffix = "self" if node.phase() in _SELF_SESSION_PHASES else "plan"
    return f"{node.id}:{suffix}"


def _brief_obj(obj) -> str:
    for field in ("command", "path", "file_path", "pattern", "title", "message"):
        value = getattr(obj, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\n", " ")[:120]
    for field in ("text", "content"):
        value = getattr(obj, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\n", " ")[:120]
    return ""


class _StepCounter:
    """数步，顺便留一份最近上下文，给审计看。"""

    def __init__(self, node_id: str, sink=None) -> None:
        self.n = 0
        self._node_id = node_id
        self._sink = sink
        self.records: list[tuple[int, str]] = []

    def __call__(self, event) -> None:
        if isinstance(event, ActionEvent):
            self.n += 1
            tool = getattr(event, "tool_name", "?")
            what = _brief_obj(getattr(event, "action", None))
            self.records.append((self.n, f"{self.n}. action {tool}: {what}".strip()))
            if self._sink is not None:
                self._sink(
                    json.dumps(
                        {
                            "node": self._node_id,
                            "step": self.n,
                            "tool": tool,
                            "what": what,
                        },
                        ensure_ascii=False,
                    )
                )
            return

        what = _brief_obj(getattr(event, "observation", event))
        if what:
            self.records.append(
                (self.n, f"{self.n}. observation {event.__class__.__name__}: {what}")
            )

    def since(self, start: int) -> str:
        return "\n".join(text for step, text in self.records if step > start)


def _message_text(event: MessageEvent) -> str:
    parts: list[str] = []
    for block in event.llm_message.content or []:
        if isinstance(block, TextContent) and block.text.strip():
            parts.append(block.text.strip())
    return "\n".join(parts)


def _node_runtime(node: Node, condense_every: int, *, llm=None):
    is_self_work = node.phase() in _SELF_SESSION_PHASES
    if is_self_work:
        tools = coding_tools() + self_work_decision_tools()
        system = self_work_system()
    else:
        tools = readonly_tools() + plan_decision_tools(node.phase())
        system = plan_system()

    agent = Agent(
        llm=llm or make_llm(),
        tools=tools,
        agent_context=AgentContext(system_message_suffix=system),
        condenser=make_condenser(max_size=condense_every) if condense_every else None,
    )
    return agent


def append_context_message(
    node: Node,
    workspace: Path,
    message: str,
    *,
    run_id: str,
    condense_every: int = 240,
) -> None:
    """Append a lightweight message to an existing node conversation."""
    session_key = _session_key(node)
    conversation = Conversation(
        agent=_node_runtime(node, condense_every),
        workspace=str(workspace),
        persistence_dir=str(persistence_dir_for(workspace, run_id, session_key)),
        conversation_id=conversation_id_for(run_id, session_key),
        visualizer=None,
        callbacks=[],
        max_iteration_per_run=_UNBOUNDED_ITERATION_BUDGET,
        stuck_detection=True,
        delete_on_close=False,
    )
    try:
        conversation.send_message(message)
    finally:
        try:
            conversation.close()
        except Exception:
            pass


def run_turn(
    tree: Tree,
    node: Node,
    workspace: Path,
    *,
    llm=None,
    on_event=None,
    step_sink=None,
    condense_every: int = 240,
    max_iterations: Optional[int] = None,
    run_id: str,
) -> TurnResult:
    session_key = _session_key(node)
    resume = has_persisted_session(workspace, run_id, session_key)
    task = turn_message(tree, node, resume=resume, project_root=str(workspace))

    counter = _StepCounter(node.id, step_sink)
    callbacks = [counter] if on_event is None else [counter, on_event]
    texts: list[str] = []

    def _keep_text(event) -> None:
        if isinstance(event, MessageEvent) and event.source == "agent":
            text = _message_text(event)
            if text:
                texts.append(text)

    callbacks.append(_keep_text)
    iteration_budget = (
        max_iterations
        if max_iterations is not None and max_iterations > 0
        else _UNBOUNDED_ITERATION_BUDGET
    )

    agent = _node_runtime(node, condense_every, llm=llm)
    conversation = Conversation(
        agent=agent,
        workspace=str(workspace),
        persistence_dir=str(persistence_dir_for(workspace, run_id, session_key)),
        conversation_id=conversation_id_for(run_id, session_key),
        visualizer=None,
        callbacks=callbacks,
        max_iteration_per_run=iteration_budget,
        stuck_detection=True,
        delete_on_close=False,
    )

    note = ""
    decision: Optional[Decision] = None
    try:
        if task:
            conversation.send_message(task)
        conversation.max_iteration_per_run = iteration_budget
        conversation.run()
        decision = take_decision(conversation)
        if decision is None:
            note = "配额用尽或未交回决定"
    finally:
        try:
            conversation.close()
        except Exception:
            pass

    return TurnResult(
        decision=decision,
        steps=counter.n,
        note=note,
        text=texts[-1] if texts else "",
        resumed=resume,
    )
