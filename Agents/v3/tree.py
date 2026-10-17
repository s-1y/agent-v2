"""工作流树。

V3 的核心语义：
- 任务节点先经过阶段自动机；分工节点写不可变 plan，一个 plan 对应一个 workspace。
- 节点组先创建、后 dispatch；dispatch 把完整 plan 分配给成员。
- progress 不再由模型手写，当前状态从 workspace / group / question 派生。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

NodeType = Literal["agent"]
Status = Literal["staged", "running", "waiting", "finished"]
WorkMode = Literal["self", "delegate"]
OrgMode = Literal["sequence", "parallel"]
WorkspaceStatus = Literal[
    "choosing_work_mode",
    "choosing_org_mode",
    "planning",
    "planning_self",
    "summarizing_self",
    "designing_sequence",
    "designing_parallel",
    "planning_sequence",
    "planning_parallel",
    "planned",
    "group_created",
    "dispatched",
    "review",
    "closed",
]
GroupKind = Literal["parallel", "sequence"]
GroupStatus = Literal["staged", "dispatched", "closed"]


class ChildSlot(BaseModel):
    """workspace 里一个计划槽位和实际 child node 的绑定。"""

    title: str
    node_id: str
    node_type: NodeType = "agent"
    returned: bool = False


class NodeGroup(BaseModel):
    """当前 workspace 创建出来的一组节点。"""

    kind: GroupKind
    count: int
    status: GroupStatus = "staged"
    slots: list[ChildSlot] = Field(default_factory=list)
    current_index: int = 0


class ChildQuestion(BaseModel):
    """组员干活时向上级提出的问题。"""

    id: str
    child_id: str
    question: str


class ChildReport(BaseModel):
    """组员返回给当前 workspace 的最新报告。"""

    child_id: str
    title: str
    message: str


class Workspace(BaseModel):
    """一个不可变 plan 工作区。"""

    id: str
    failure_summary: str = ""
    plan: str = ""
    position: str = ""  # harness 生成的短路标，只用于给新成员一次性说明树中位置
    status: WorkspaceStatus = "planning"
    work_mode: Optional[WorkMode] = None
    org_mode: Optional[OrgMode] = None
    group: Optional[NodeGroup] = None
    question_seq: int = 0
    questions: list[ChildQuestion] = Field(default_factory=list)
    child_reports: list[ChildReport] = Field(default_factory=list)


class Node(BaseModel):
    id: str
    parent_id: Optional[str] = None
    node_type: NodeType = "agent"
    title: str = ""

    # 共有字段
    from_parent: str = ""
    position: str = ""  # harness 生成的短定位，避免逐层展开祖先 plan
    status: Status = "running"

    # 任务节点专有
    children_ids: list[str] = Field(default_factory=list)
    workspaces: list[Workspace] = Field(default_factory=list)

    # 最近一次向上级返回的话
    return_message: str = ""

    @field_validator("node_type", mode="before")
    @classmethod
    def _coerce_legacy_language(cls, value: Any) -> Any:
        # 旧树 dumps 里曾用 language/plan/code，新系统统一视为 agent。
        if value in {"language", "plan", "code"}:
            return "agent"
        return value

    def active_workspace(self) -> Optional[Workspace]:
        for workspace in reversed(self.workspaces):
            if workspace.status != "closed":
                return workspace
        return None

    def next_workspace_id(self) -> str:
        return f"w{len(self.workspaces) + 1}"

    def open_question(self) -> Optional[ChildQuestion]:
        workspace = self.active_workspace()
        if workspace is None:
            return None
        return workspace.questions[0] if workspace.questions else None

    def phase(self) -> str:
        question = self.open_question()
        if question is not None:
            return "7_answer_question"
        workspace = self.active_workspace()
        if workspace is None:
            return "1_orient"
        if workspace.status == "choosing_work_mode":
            return "2_choose_work_mode"
        if workspace.status == "choosing_org_mode":
            return "3B_choose_org_mode"
        if workspace.status == "planning_self":
            return "3A_self_work"
        if workspace.status == "summarizing_self":
            return "3A2_summarize"
        if workspace.status == "designing_sequence":
            return "4A1_design_sequence"
        if workspace.status == "planning_sequence":
            return "4A2_write_sequence_plan"
        if workspace.status == "designing_parallel":
            return "4B1_design_parallel"
        if workspace.status == "planning_parallel":
            return "4B2_write_parallel_plan"
        if workspace.status == "planning":
            return "write_plan"
        if workspace.status == "planned":
            return "5_create_group"
        if workspace.status == "group_created":
            return "6_dispatch"
        if workspace.status == "review":
            return "8_review"
        return "waiting"


class Tree(BaseModel):
    root_id: str = "n0"
    root_brief: str = ""  # 人给的要求，就是根的 from_parent
    nodes: dict[str, Node] = Field(default_factory=dict)
    seq: int = 0

    # --- 基本操作 ---

    def new_id(self) -> str:
        self.seq += 1
        return f"n{self.seq}"

    @property
    def root(self) -> Node:
        return self.nodes[self.root_id]

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def from_parent(self, node: Node) -> str:
        """组员读到的是创建 / dispatch 时写入自己的 from_parent 快照。"""
        return node.from_parent or (self.root_brief if node.parent_id is None else "")

    def open_children(self, node: Node) -> list[str]:
        """还在实际运行或等待回答的成员；staged 只是占位，不阻塞上级。"""
        return [
            i
            for i in node.children_ids
            if self.nodes[i].status in {"running", "waiting"}
        ]

    def depth(self, node: Node) -> int:
        d = 0
        cur = node
        while cur.parent_id is not None:
            cur = self.nodes[cur.parent_id]
            d += 1
        return d

    # --- 生长 ---

    def create_root(self, brief: str) -> Node:
        self.root_brief = brief
        root = Node(
            id=self.root_id,
            title="root",
            from_parent=brief,
        )
        self.nodes[root.id] = root
        return root

    def add_child(
        self,
        parent: Node,
        *,
        node_type: NodeType,
        title: str,
        from_parent: str = "",
        status: Status = "staged",
    ) -> Node:
        child = Node(
            id=self.new_id(),
            parent_id=parent.id,
            node_type=node_type,
            title=title,
            from_parent=from_parent,
            status=status,
        )
        self.nodes[child.id] = child
        parent.children_ids.append(child.id)
        return child

    # --- 落盘 ---

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Tree":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
