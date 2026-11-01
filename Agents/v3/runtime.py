"""跑一个节点的一个回合：起会话、装工具、数步、拿回决定。

同一次运行里，每个 node_id 对应一份可挂起的 OpenHands 会话：
    <workspace>/.meta_agent/runs/<run_id>/conversations/<node_id>:plan/...
自己做不再另开会话。回合结束 close() 释放工具；磁盘保留。同一次运行里再次开工按同一 node_id resume，
只追加一条继续消息，不再冷启动整页任务卡。另一次运行用另一个 run_id，从空会话开始。
发给 OpenHands 的 workspace 仍是项目根目录，读写的是那里的文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Agents.engine import (
    release_browser,
    ActionEvent,
    Agent,
    AgentContext,
    Conversation,
    FileEditorTool,
    MessageEvent,
    TaskTrackerTool,
    TerminalTool,
    TextContent,
    Tool,
    make_condenser,
    make_llm,
    readonly_tools,
    result_view_tools,
)
from Agents.v3.context import (
    plan_system,
    root_system,
    turn_message,
)
from Agents.v3.session import (
    conversation_id_for,
    has_persisted_session,
    persistence_dir_for,
)
from Agents.v3.tools import (
    Decision,
    designer_decision_tools,
    plan_decision_tools,
    root_decision_tools,
    reviewer_decision_tools,
    take_decision,
)
from Agents.v3.tree import Designer, Node, Reviewer, Root, Tree

_UNBOUNDED_ITERATION_BUDGET = 1_000_000


@dataclass
class TurnResult:
    decision: Optional[Decision]
    steps: int
    note: str = ""
    text: str = ""
    resumed: bool = False
    error: bool = False


def _session_key(node: Node) -> str:
    return f"{node.id}:plan"


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


def _agent_turn_text(event) -> str:
    """这一回合里写下的话。普通回复算，finish 里的结论也算。"""
    if isinstance(event, MessageEvent) and event.source == "agent":
        return _message_text(event)
    if not isinstance(event, ActionEvent) or event.tool_name != "finish":
        return ""
    action = event.action
    message = getattr(action, "message", "") if action is not None else ""
    if isinstance(message, str) and message.strip():
        return message.strip()
    parts: list[str] = []
    for block in event.thought or []:
        text = getattr(block, "text", "")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def _worker_tools(node: Node) -> list[Tool]:
    """认任务时的工具不删。自己做只加上 terminal、file_editor、task_tracker。

    一旦加上就不能再拿掉，所以概括、交用法也留着。只分工、自己没写代码的人不会加上。
    """
    tools = readonly_tools() + plan_decision_tools(node.phase())
    workspace = node.active_workspace()
    if workspace is not None and workspace.work_mode == "self":
        tools = tools + [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]
    return tools


def _node_runtime(node: Node, condense_every: int, *, llm=None):
    if isinstance(node, Designer):
        # 决定工具从第一轮就给全。写文件的三件套从写手册起加上，之后不能拿掉：
        # 对接验收仍是同一段对话，少了 terminal / file_editor / task_tracker 会无法恢复。
        tools = readonly_tools() + designer_decision_tools(node.phase())
        workspace = node.active_workspace()
        still_reading_requirement = (
            workspace is not None and workspace.status == "designer_requirement"
        )
        if workspace is not None and not still_reading_requirement:
            tools = tools + [
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
            ]
        system = plan_system(node)
    elif isinstance(node, Reviewer):
        tools = result_view_tools() + reviewer_decision_tools(node.phase())
        system = plan_system(node)
    elif isinstance(node, Root):
        tools = readonly_tools() + root_decision_tools(node.phase())
        system = root_system()
    else:
        tools = _worker_tools(node)
        system = plan_system(node)

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
    task = turn_message(
        tree, node, resume=resume, project_root=str(workspace), run_id=run_id
    )

    counter = _StepCounter(node.id, step_sink)
    callbacks = [counter] if on_event is None else [counter, on_event]
    texts: list[str] = []

    def _keep_text(event) -> None:
        text = _agent_turn_text(event)
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
        release_browser(conversation)

    return TurnResult(
        decision=decision,
        steps=counter.n,
        note=note,
        text=texts[-1] if texts else "",
        resumed=resume,
    )
