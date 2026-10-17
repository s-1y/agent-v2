"""把节点的 workspace 状态填进 prompts/ 下的模板。"""

from __future__ import annotations

from pathlib import Path

from Agents.v3.tree import Node, Tree

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PHASE_PROMPT_DIR = PROMPT_DIR / "phases"

_EMPTY = "（还没有）"
_STATE_CAP = 6000


def load(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def load_phase(name: str) -> str:
    path = PHASE_PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def fill(name: str, **values: str) -> str:
    text = load(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value or _EMPTY)
    return text


def _render_chain_raw(tree: Tree, node: Node) -> str:
    if node.parent_id is None:
        return "这是整件事的起点，你就是根。"

    links: list[tuple[str, str]] = []
    cur = node
    while cur.parent_id is not None:
        parent = tree.get(cur.parent_id)
        workspace = parent.active_workspace()
        plan_head = ""
        if workspace and workspace.plan:
            plan_head = workspace.position or workspace.plan.strip().splitlines()[0]
        links.append((plan_head, cur.title))
        cur = parent
    links.reverse()

    lines = []
    for i, (split_into, you_are) in enumerate(links, start=1):
        parent_bit = (split_into or "").strip() or "（上级尚未写下概要）"
        lines.append(
            f"第 {i} 层：上级这样安排——{parent_bit}\n"
            f"        在此基础上，你负责：{you_are}"
        )
    return "\n".join(lines)


def render_chain_source(tree: Tree, node: Node) -> str:
    """给定位压缩模型的原料：每一层上级的完整 plan，加上下级负责哪一块。"""
    if node.parent_id is None:
        return _render_chain_raw(tree, node)

    layers: list[tuple[str, str]] = []
    cur = node
    while cur.parent_id is not None:
        parent = tree.get(cur.parent_id)
        workspace = parent.active_workspace()
        plan = workspace.plan.strip() if workspace and workspace.plan.strip() else ""
        layers.append((plan or "（上级尚未写下 plan）", cur.title))
        cur = parent
    layers.reverse()

    blocks = []
    last = len(layers)
    for i, (plan, title) in enumerate(layers, start=1):
        who = "当前成员" if i == last else "下一层成员"
        blocks.append(
            f"### 第 {i} 层上级的 plan\n{plan}\n\n"
            f"在这一层，{who}负责上面 plan 里的「{title}」。"
        )
    return "\n\n".join(blocks)


def render_chain(tree: Tree, node: Node) -> str:
    if node.position.strip():
        return node.position.strip()
    return _render_chain_raw(tree, node)


def _workspace_state(tree: Tree, node: Node, *, include_plan: bool = True) -> str:
    workspace = node.active_workspace()
    if workspace is None:
        return "当前还没有工作区。请先理解任务并 write_plan。"

    lines = [
        f"工作区：{workspace.id}",
        f"阶段：{workspace.status}",
    ]
    if workspace.work_mode:
        lines.append(f"工作方式：{workspace.work_mode}")
    if workspace.org_mode:
        lines.append(f"组织方式：{workspace.org_mode}")
    if workspace.failure_summary.strip():
        lines.append(f"之前失败总结：{workspace.failure_summary}")

    if workspace.questions:
        lines.append("")
        lines.append("待回答问题：")
        for question in workspace.questions:
            lines.append(f"- {question.id} from {question.child_id}: {question.question}")

    if workspace.child_reports:
        lines.append("")
        lines.append("组员返回报告：")
        for report in workspace.child_reports:
            lines.append(f"- {report.child_id} {report.title}: {report.message}")

    if workspace.group is not None:
        group = workspace.group
        lines.append("")
        lines.append(
            f"节点组：{group.kind} / {group.status} / count={group.count} / current={group.current_index + 1}"
        )
        for i, slot in enumerate(group.slots, start=1):
            child = tree.get(slot.node_id)
            returned = "returned" if slot.returned else "not_returned"
            lines.append(
                f"- {i}. {slot.node_id} {child.node_type} 子任务 {i} {child.title}："
                f"{child.status}, {returned}"
            )

    if include_plan and workspace.plan.strip():
        lines.append("")
        lines.append("Plan：")
        lines.append(workspace.plan.strip())

    text = "\n".join(lines)
    if len(text) > _STATE_CAP:
        text = text[:_STATE_CAP] + "\n……（workspace 状态过长，已截断）"
    return text


def _common(
    tree: Tree, node: Node, *, project_root: str = "", include_plan: bool = True
) -> dict[str, str]:
    phase = node.phase()
    return {
        "position": render_chain(tree, node),
        "project_root": project_root,
        "from_parent": tree.from_parent(node),
        "phase": phase,
        "phase_prompt": load_phase(phase),
        "workspace_state": _workspace_state(
            tree, node, include_plan=include_plan
        ),
    }


def _plan_message(
    tree: Tree, node: Node, *, resume: bool, project_root: str = ""
) -> str:
    values = _common(tree, node, project_root=project_root, include_plan=not resume)
    title = "继续" if resume else "开工"
    task = (
        ""
        if resume
        else f"""## 上级交给你的任务

{values["from_parent"] or _EMPTY}

"""
    )
    position = (
        f"""## 你在整件事里的位置

{values["position"] or _EMPTY}

"""
        if not resume
        else ""
    )
    root = (
        f"""## 项目根目录

{values["project_root"] or _EMPTY}

操作请在这里进行。

"""
        if not resume
        else ""
    )
    return f"""## {title}

你是同一个任务节点。请根据当前阶段行动。

{position}{root}{task}## 当前阶段

{values["phase"] or _EMPTY}

## 阶段提示

{values["phase_prompt"] or _EMPTY}

## 当前工作区状态

{values["workspace_state"] or _EMPTY}
""".strip()


def plan_task(tree: Tree, node: Node, *, project_root: str = "") -> str:
    return _plan_message(tree, node, resume=False, project_root=project_root)


def plan_continue(tree: Tree, node: Node) -> str:
    """唤醒已有会话时只追加阶段提示。任务和 plan 已在历史里，不再重贴。"""
    return _plan_message(tree, node, resume=True)


def self_work_continue(tree: Tree, node: Node) -> str:
    return ""


def turn_message(
    tree: Tree, node: Node, *, resume: bool, project_root: str = ""
) -> str:
    if resume:
        return (
            self_work_continue(tree, node)
            if node.phase() == "3A_self_work"
            else plan_continue(tree, node)
        )
    return plan_task(tree, node, project_root=project_root)

def plan_system() -> str:
    return load_phase("0_system")


def self_work_system() -> str:
    return load_phase("0_system")
