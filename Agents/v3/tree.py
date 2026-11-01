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

from pydantic import BaseModel, Field, model_validator

NodeType = Literal["agent", "root", "designer", "reviewer"]
Status = Literal["staged", "running", "waiting", "finished"]
WorkMode = Literal["self", "delegate"]
OrgMode = Literal["sequence", "parallel"]
WorkspaceStatus = Literal[
    "choosing_work_mode",
    "choosing_org_mode",
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
    "worker_final",
    "reviewer_look",
    "reviewer_plan",
    "reviewer_dispatch",
    "reviewer_filter",
    "member_check",
    "member_return",
    "designer_revise_look",
    "designer_revise",
    "root_look",
    "root_send_doc",
    "root_wait_doc",
    "root_send_build",
    "root_wait_build",
    "root_send_review",
    "root_wait_review",
    "root_send_revise",
    "root_wait_usage",
    "designer_requirement",
    "designer_write",
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
    status: WorkspaceStatus = "choosing_work_mode"
    work_mode: Optional[WorkMode] = None
    org_mode: Optional[OrgMode] = None
    group: Optional[NodeGroup] = None
    question_seq: int = 0
    questions: list[ChildQuestion] = Field(default_factory=list)
    child_reports: list[ChildReport] = Field(default_factory=list)
    inbox: str = ""  # 系统按顺序拼好、交回负责人的核验记录


class Node(BaseModel):
    id: str
    parent_id: Optional[str] = None
    node_type: NodeType = "agent"
    title: str = ""

    # 共有字段
    from_parent: str = ""
    position: str = ""  # harness 生成的短定位，避免逐层展开祖先 plan
    status: Status = "running"
    order: int = 0  # 创建顺序，用来在同一层里排先后
    review_role: Literal["", "leader", "member"] = ""

    # 任务节点专有
    children_ids: list[str] = Field(default_factory=list)
    workspaces: list[Workspace] = Field(default_factory=list)

    # 最近一次向上级返回的话
    return_message: str = ""

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
        if workspace.status == "worker_final":
            return "9_worker_final"
        if workspace.status == "planned":
            return "5_create_group"
        if workspace.status == "group_created":
            return "6_dispatch"
        if workspace.status == "review":
            return "8_review"
        return "waiting"


class Root(Node):
    """听人的原话，派设计、开发、验收。不写计划，不自己做。"""

    node_type: Literal["root"] = "root"

    def phase(self) -> str:
        if self.open_question() is not None:
            return "7_answer_question"
        workspace = self.active_workspace()
        if workspace is None:
            return "root_1_look"
        status = workspace.status
        if status == "root_look":
            return "root_1_look"
        if status == "root_send_doc":
            return "root_2_send_doc"
        if status == "root_send_build":
            return "root_3_send_build"
        if status == "root_send_review":
            return "root_6_send_review"
        if status == "root_send_revise":
            return "root_7_send_revise"
        return "waiting"


class Designer(Node):
    """写内部计划文档的人。先补全要求，再把文档写到 .meta_agent。"""

    node_type: Literal["designer"] = "designer"

    def phase(self) -> str:
        if self.open_question() is not None:
            return "7_answer_question"
        workspace = self.active_workspace()
        if workspace is None or workspace.status == "designer_requirement":
            return "1_complete_requirement"
        if workspace.status == "designer_write":
            return "2_write_plan"
        if workspace.status == "designer_revise_look":
            return "1A_read_review"
        if workspace.status == "designer_revise":
            return "2A_write_revision"
        return "waiting"


class Reviewer(Node):
    """验收。负责人划分并筛选；队员只核验自己那一部分，不再往下派。"""

    node_type: Literal["reviewer"] = "reviewer"

    def phase(self) -> str:
        if self.open_question() is not None and self.review_role == "leader":
            return "leader_4_answer"
        workspace = self.active_workspace()
        if self.review_role == "member":
            if workspace is None or workspace.status == "member_check":
                return "member_check"
            if workspace.status == "member_return":
                return "member_return"
            return "waiting"
        if workspace is None or workspace.status == "reviewer_look":
            return "leader_1_look"
        if workspace.status == "reviewer_plan":
            return "leader_2_plan"
        if workspace.status == "reviewer_dispatch":
            return "leader_3_dispatch"
        if workspace.status == "reviewer_filter":
            return "leader_5_filter"
        return "waiting"


class Tree(BaseModel):
    root_id: str = "Root"
    root_brief: str = ""  # 人给的要求，就是根的 from_parent
    nodes: dict[str, Node] = Field(default_factory=dict)
    seq: int = 0
    plan_version: int = 1
    delivery: str = ""  # 开发负责人交回的用法
    review_note: str = ""  # 验收负责人筛选后的改进建议
    usage_for_version: int = 0  # 已经问过用法的计划版本

    # --- 基本操作 ---

    def display_name(self, node_type: NodeType) -> str:
        """对外名字。Root 和 Plan 各只有一个，不编号。开发从 Worker 0 起。"""
        if node_type == "root":
            return "Root"
        if node_type == "designer":
            return "Plan"
        if node_type == "agent":
            taken = sum(1 for node in self.nodes.values() if node.node_type == "agent")
            return f"Worker {taken}"
        taken = sum(1 for node in self.nodes.values() if node.node_type == "reviewer")
        if taken == 0:
            return "Reviewer"
        return f"Reviewer {taken - 1}"

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

    def create_root(self, brief: str) -> Root:
        self.root_brief = brief
        root = Root(
            id=self.root_id,
            title="Root",
            from_parent=brief,
            workspaces=[
                Workspace(id="w1", status="root_look"),
            ],
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
        kind = {
            "designer": Designer,
            "reviewer": Reviewer,
            "root": Root,
        }.get(node_type, Node)
        name = self.display_name(node_type)
        self.seq += 1
        child = kind(
            id=name,
            parent_id=parent.id,
            title=name,
            order=self.seq,
            from_parent=from_parent,
            status=status,
            review_role=(
                "member"
                if node_type == "reviewer" and isinstance(parent, Reviewer)
                else "leader"
                if node_type == "reviewer"
                else ""
            ),
        )
        if isinstance(child, Designer):
            child.workspaces.append(Workspace(id="w1", status="designer_requirement"))
        elif isinstance(child, Reviewer):
            initial = "member_check" if child.review_role == "member" else "reviewer_look"
            child.workspaces.append(Workspace(id="w1", status=initial))
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

    @model_validator(mode="before")
    @classmethod
    def _keep_root_class(cls, data: Any) -> Any:
        """node_type 为 root 时读回 Root，不要收成普通节点。"""
        if not isinstance(data, dict):
            return data
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return data
        parsed: dict[str, Any] = {}
        for key, value in nodes.items():
            if isinstance(value, Node):
                parsed[key] = value
            elif isinstance(value, dict) and value.get("node_type") == "root":
                parsed[key] = Root.model_validate(value)
            elif isinstance(value, dict) and value.get("node_type") == "designer":
                parsed[key] = Designer.model_validate(value)
            elif isinstance(value, dict) and value.get("node_type") == "reviewer":
                parsed[key] = Reviewer.model_validate(value)
            else:
                parsed[key] = value
        data = dict(data)
        data["nodes"] = parsed
        return data

    @classmethod
    def load(cls, path: Path) -> "Tree":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
