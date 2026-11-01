"""主循环：挑一个可跑的节点，跑一个回合，按决定推进 workspace 状态机。"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

from Agents.v3.config import Config
from Agents.v3.context import _render_chain_raw, render_chain_source
from Agents.v3.runtime import TurnResult, _session_key, append_context_message, run_turn
from Agents.v3.session import has_persisted_session, plan_doc_path, plan_docs_ready
from Agents.v3.tools import Decision
from Agents.v3.tree import (
    ChildQuestion,
    ChildReport,
    ChildSlot,
    Designer,
    Node,
    NodeGroup,
    Reviewer,
    Root,
    Tree,
    Workspace,
    WorkspaceStatus,
)

_ROOT_AFTER_RETURN = {
    "designer": "root_send_build",
    "agent": "root_send_review",
    "reviewer": "root_send_revise",
}


def _summarize_position(plan: str) -> str:
    """Harness-owned short label for tree navigation.

    Keep this deterministic and side-effect free. A cheap LLM summarizer can
    replace this function later without changing the state machine.
    """
    for raw in plan.splitlines():
        line = raw.strip().lstrip("#").strip()
        if line:
            return line[:120]
    return ""


def _position_summary(cfg: Config, plan: str) -> str:
    summarize = cfg.position_summarizer or _summarize_position
    try:
        text = summarize(plan).strip()
    except Exception:
        text = ""
    return text or _summarize_position(plan)


def _summary_note(summary: str) -> str:
    return f"；summary: {summary}" if summary else ""


def _summarize_child_position(tree: Tree, child: Node, cfg: Config) -> str:
    raw = _render_chain_raw(tree, child)
    summarize = cfg.chain_summarizer
    if summarize is None:
        return raw
    try:
        text = summarize(render_chain_source(tree, child)).strip()
    except Exception:
        text = ""
    return text or raw


def _workspace_label(node: Node, workspace: Workspace) -> str:
    return f"{node.id}/{workspace.id}"


def runnable(tree: Tree) -> list[Node]:
    """status 是 running，且当前 phase 有可用动作的节点。"""
    return [
        n
        for n in tree.nodes.values()
        if n.status == "running" and n.phase() != "waiting"
    ]


def _schedule_key(tree: Tree, node: Node) -> tuple[int, int, int]:
    return (
        0 if node.open_question() else 1,
        tree.depth(node),
        node.order,
    )


_SUMMARY_ITERATIONS = 4
_SUMMARY_FALLBACK = "停工了，但没有写出总结。"


def _is_self_work(node: Node) -> bool:
    return node.phase() in {"3A_self_work", "3A2_summarize"}


def _max_iterations_for(node: Node, cfg: Config) -> int | None:
    phase = node.phase()
    if phase == "3A2_summarize":
        return _SUMMARY_ITERATIONS
    if phase == "3A_self_work":
        return cfg.max_self_iterations
    return None


def select_batch(tree: Tree, ready: list[Node], cfg: Config) -> list[Node]:
    """Choose a batch: many planning nodes, at most a few self-work nodes."""
    plan_nodes = sorted(
        [node for node in ready if not _is_self_work(node)],
        key=lambda node: _schedule_key(tree, node),
    )
    self_work_nodes = sorted(
        [node for node in ready if _is_self_work(node)],
        key=lambda node: _schedule_key(tree, node),
    )
    batch: list[Node] = []
    if cfg.max_parallel_plan > 0:
        batch.extend(plan_nodes[: cfg.max_parallel_plan])
    if cfg.max_parallel_self_work > 0:
        batch.extend(self_work_nodes[: cfg.max_parallel_self_work])
    return batch


def _render_assignment(parent: Node, workspace: Workspace, task_index: int) -> str:
    failure = (
        f"\n## 之前工作区失败总结\n{workspace.failure_summary}\n"
        if workspace.failure_summary.strip()
        else ""
    )
    return (
        f"你来自父节点 {parent.id} 的工作区 {workspace.id}。\n"
        f"你负责 plan 里的第 {task_index} 个子任务。\n"
        "完整 plan 是上下文；你自己阅读 plan，找到第 "
        f"{task_index} 个子任务并只推进这一块，不要做其他成员的任务。\n"
        f"{failure}\n"
        f"## 完整 plan\n{workspace.plan}"
    ).strip()


def _active_workspace(node: Node) -> Workspace | None:
    return node.active_workspace()


def _require_workspace(node: Node) -> tuple[Workspace | None, str | None]:
    workspace = _active_workspace(node)
    if workspace is None:
        return None, f"{node.id} 当前没有工作区"
    return workspace, None


def _slot_for_child(workspace: Workspace, child_id: str) -> ChildSlot | None:
    if workspace.group is None:
        return None
    for slot in workspace.group.slots:
        if slot.node_id == child_id:
            return slot
    return None


def _active_group_children(tree: Tree, workspace: Workspace) -> list[Node]:
    if workspace.group is None:
        return []
    return [
        tree.get(slot.node_id)
        for slot in workspace.group.slots
        if tree.get(slot.node_id).status in {"running", "waiting"}
    ]


def _all_dispatched_slots_returned(tree: Tree, workspace: Workspace) -> bool:
    if workspace.group is None:
        return False
    for slot in workspace.group.slots:
        child = tree.get(slot.node_id)
        if child.status != "staged" and not slot.returned:
            return False
    return not _active_group_children(tree, workspace)


def _next_question_id(workspace: Workspace) -> str:
    workspace.question_seq += 1
    return f"q{workspace.question_seq}"


_DESIGN_NEXT_STATUS: dict[str, WorkspaceStatus] = {
    "designing_sequence": "planning_sequence",
    "designing_parallel": "planning_parallel",
}


def _allowed_decisions(node: Node) -> set[str] | None:
    """Decision tools allowed by the current prompt stage; None = unrestricted."""
    phase = node.phase()
    allowed_by_phase: dict[str, set[str]] = {
        "1_orient": set(),
        "2_choose_work_mode": {"choose_work_mode"},
        "3A_self_work": {"ask_parent"},
        "3A2_summarize": set(),
        "3B_choose_org_mode": {"choose_org_mode"},
        "4A1_design_sequence": set(),
        "4A2_write_sequence_plan": {"write_plan", "ask_parent"},
        "4B1_design_parallel": set(),
        "4B2_write_parallel_plan": {"write_plan", "ask_parent"},
        "5_create_group": {"create_sequence_group", "create_parallel_group"},
        "6_dispatch": {"dispatch"},
        "7_answer_question": {"answer_child_question"},
        "8_review": {"return_to_parent", "continue_child", "change_plan", "ask_parent"},
        "9_worker_final": {"return_to_parent"},
        "leader_1_look": set(),
        "leader_2_plan": set(),
        "leader_3_dispatch": {"send_review_plan"},
        "leader_4_answer": {"answer_child_question"},
        "leader_5_filter": {"return_to_parent"},
        "member_check": {"ask_parent"},
        "member_return": {"return_to_parent"},
        "root_1_look": set(),
        "root_2_send_doc": {"send_next"},
        "root_3_send_build": {"send_next"},
        "root_6_send_review": {"send_next", "continue_child"},
        "root_7_send_revise": {"send_next"},
        "1_complete_requirement": set(),
        "2_write_plan": {"finish_design"},
        "1A_read_review": set(),
        "2A_write_revision": {"finish_design"},
    }
    return allowed_by_phase.get(phase)


def _allowed_text(allowed: set[str]) -> str:
    """这一阶段的决策工具提示。空集合表示只用文字，不列工具名。"""
    if not allowed:
        return "不调用决策工具，直接用文字回答"
    return "只能调用 " + "、".join(sorted(allowed))


def _decision_error_feedback(node: Node, decision: Decision) -> str | None:
    allowed = _allowed_decisions(node)
    if allowed is None or decision.kind in allowed:
        return None
    return (
        "## 外层状态机拒绝了上次工具调用\n\n"
        f"你刚才调用了 `{decision.kind}`，但当前阶段是 `{node.phase()}`。\n\n"
        f"当前阶段{_allowed_text(allowed)}。\n\n"
        "请根据当前阶段提示重新行动。"
    )


def _begin_summary(node: Node) -> bool:
    """写代码停工后，留在同一会话里概括，不把半句话直接交给上级。"""
    if node.phase() != "3A_self_work":
        return False
    workspace = node.active_workspace()
    if workspace is None or workspace.status != "planning_self":
        return False
    workspace.status = "summarizing_self"
    node.status = "running"
    return True


def _finish_summary(tree: Tree, node: Node, text: str, cfg: Config) -> str:
    node.return_message = text.strip() or _SUMMARY_FALLBACK
    if _begin_usage_report(tree, node):
        return f"{node.id} 请交回用法，不要写代码"
    node.status = "finished"
    if node.parent_id is not None:
        _wake_parent(tree, node, cfg)
        return f"{node.id} 总结完成，返回"
    return f"{node.id} 总结完成"


def _begin_usage_report(tree: Tree, node: Node) -> bool:
    """开发负责人交回后，先追问用法，再把这一轮交给根。"""
    if node.parent_id is None or node.node_type != "agent":
        return False
    parent = tree.get(node.parent_id)
    if not isinstance(parent, Root):
        return False
    if tree.usage_for_version >= tree.plan_version:
        return False
    workspace = node.active_workspace()
    if workspace is None or workspace.status == "worker_final":
        return False
    workspace.status = "worker_final"
    node.status = "running"
    node.return_message = ""
    return True


def _collect_review_if_ready(tree: Tree, parent: Reviewer, workspace: Workspace) -> None:
    """队员都交回、也没有待答问题之后，把记录按顺序拼给负责人。"""
    if parent.open_question() is not None:
        parent.status = "running"
        return
    group = workspace.group
    if group is None or not _all_dispatched_slots_returned(tree, workspace):
        parent.status = "waiting"
        return
    if workspace.status != "reviewer_filter":
        workspace.inbox = _bundle_member_reports(workspace)
        workspace.status = "reviewer_filter"
    parent.status = "running"


def _bundle_member_reports(workspace: Workspace) -> str:
    group = workspace.group
    if group is None:
        return ""
    by_id = {report.child_id: report for report in workspace.child_reports}
    parts: list[str] = []
    for index, slot in enumerate(group.slots, start=1):
        report = by_id.get(slot.node_id)
        if report is None:
            body = "（没有返回内容）"
        elif not report.message.strip():
            body = "（这一部分没有要提交的条目）"
        else:
            body = report.message.strip()
        parts.append(f"## 第 {index} 部分 {slot.title}\n\n{body}")
    return "\n\n".join(parts)


def _finish_look(node: Node, text: str) -> bool:
    """观察写完就进入下一步。没有写下内容则留在原地。"""
    if not text.strip():
        return False
    if isinstance(node, Root):
        if node.phase() != "root_1_look":
            return False
        active = node.active_workspace()
        if active is None or active.status != "root_look":
            return False
        active.status = "root_send_doc"
        node.status = "running"
        return True
    if isinstance(node, Designer):
        active = node.active_workspace()
        if node.phase() == "1_complete_requirement":
            if active is None or active.status != "designer_requirement":
                return False
            active.status = "designer_write"
            node.status = "running"
            return True
        if node.phase() == "1A_read_review":
            if active is None or active.status != "designer_revise_look":
                return False
            active.status = "designer_revise"
            node.status = "running"
            return True
        return False
    if isinstance(node, Reviewer):
        if node.phase() != "leader_1_look":
            return False
        active = node.active_workspace()
        if active is None or active.status != "reviewer_look":
            return False
        active.status = "reviewer_plan"
        node.status = "running"
        return True
    if node.phase() != "1_orient":
        return False
    active = node.active_workspace()
    if active is None:
        node.workspaces.append(
            Workspace(id=node.next_workspace_id(), status="choosing_work_mode")
        )
    elif active.status != "choosing_work_mode":
        return False
    node.status = "running"
    return True


_ROOT_SEND_PHASES = {
    "root_2_send_doc",
    "root_3_send_build",
    "root_6_send_review",
    "root_7_send_revise",
}


def _continue_root_without_decision(
    tree: Tree,
    node: Root,
    text: str,
    cfg: Config,
    workspace: Path,
) -> str:
    """根没交上决定工具时，按当前阶段该做的事继续，不结束整次运行。"""
    phase = node.phase()
    if phase == "root_1_look":
        if _finish_look(node, text.strip() or "看过了。"):
            return f"{node.id} 没调用工具，系统接着看完 -> {node.phase()}"
        node.status = "running"
        return f"{node.id} 留在 {phase}"
    if phase in _ROOT_SEND_PHASES:
        line = apply_decision(
            tree,
            node,
            Decision(kind="send_next", payload={}),
            cfg,
            workspace,
        )
        return f"{node.id} 没调用工具，系统接着办：{line}"
    if phase == "7_answer_question":
        question = node.open_question()
        if question is not None:
            line = apply_decision(
                tree,
                node,
                Decision(
                    kind="answer_child_question",
                    payload={
                        "question_id": question.id,
                        "answer": text.strip() or "你可以自己决定。",
                    },
                ),
                cfg,
                workspace,
            )
            return f"{node.id} 没调用工具，系统接着办：{line}"
    node.status = "running"
    return f"{node.id} 没给决定，留在 {phase}"


def _finish_review_plan(node: Node, text: str) -> bool:
    """验收负责人把划分想完即可。想完进入下发，这一步还不启动队员。"""
    if not isinstance(node, Reviewer) or node.review_role != "leader":
        return False
    workspace = node.active_workspace()
    if workspace is None or workspace.status != "reviewer_plan":
        return False
    if not text.strip():
        return False
    workspace.status = "reviewer_dispatch"
    node.status = "running"
    return True


def _finish_member_check(node: Node, text: str) -> bool:
    """队员把检查写完就进入归类提交，这一步还不交回。"""
    if not isinstance(node, Reviewer) or node.review_role != "member":
        return False
    if node.phase() != "member_check":
        return False
    if not text.strip():
        return False
    workspace = node.active_workspace()
    if workspace is None or workspace.status != "member_check":
        return False
    workspace.status = "member_return"
    node.status = "running"
    return True


_PART_LINE = re.compile(r"^第\s*(\d+)\s*部分", re.MULTILINE)


def _review_part_count(plan: str) -> int | None:
    """「第 1 部分」起连续编号，有几行就有几个并行队员。"""
    nums = [int(n) for n in _PART_LINE.findall(plan)]
    if not nums or nums != list(range(1, len(nums) + 1)):
        return None
    return len(nums)


_MEMBER_RETURN_TAGS = ("必须修改", "推荐调整", "备选优化")
_MEMBER_TAG_LINE = re.compile(r"^[ \t]*\[(?P<tag>[^\]\n]+)\][ \t]*(?P<rest>.*)$")


def _normalize_member_return(message: str) -> str | None:
    """收成三个可选标签。空正文合法；标签外的话、空标签、重复标签都不收。"""
    text = message.replace("\r\n", "\n").strip()
    if not text:
        return ""
    buckets: dict[str, list[str]] = {tag: [] for tag in _MEMBER_RETURN_TAGS}
    seen: set[str] = set()
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _MEMBER_TAG_LINE.match(line)
        if match is not None:
            tag = match.group("tag").strip()
            rest = match.group("rest").strip()
            if tag not in buckets:
                return None
            if tag in seen:
                return None
            seen.add(tag)
            current = tag
            if rest:
                buckets[tag].append(rest)
            continue
        if current is None:
            return None
        buckets[current].append(line)
    if seen and all(not buckets[tag] for tag in seen):
        return None
    parts: list[str] = []
    for tag in _MEMBER_RETURN_TAGS:
        items = buckets[tag]
        if not items:
            continue
        body = "\n".join(f"    {item}" for item in items)
        parts.append(f"[{tag}]\n{body}")
    return "\n\n".join(parts)


def _finish_design(node: Node) -> bool:
    """设计步不交决定即视为答完，推进到写 plan。"""
    workspace = node.active_workspace()
    if workspace is None or workspace.status not in _DESIGN_NEXT_STATUS:
        return False
    workspace.status = _DESIGN_NEXT_STATUS[workspace.status]
    node.status = "running"
    return True


def _missing_decision_feedback(node: Node, note: str) -> str:
    """这一轮没交上决定。告诉他发生了什么、现在该用哪个，留在原地。"""
    allowed = _allowed_decisions(node)
    hint = ""
    if allowed is not None:
        hint = f"\n\n当前阶段{_allowed_text(allowed)}。"
    reason = note.strip() if note and note.strip() else "这一轮没有调用当前阶段要求的决定工具。"
    return (
        "## 这一轮没有完成\n\n"
        f"{reason}"
        f"{hint}\n\n"
        "请根据当前阶段提示重新行动。"
    )


def _append_decision_error(
    node: Node, workspace: Path, message: str, cfg: Config
) -> None:
    try:
        append_context_message(
            node,
            workspace,
            message,
            run_id=cfg.run_id,
            condense_every=cfg.condense_every,
        )
    except Exception:
        pass


def _wake_parent(tree: Tree, node: Node, cfg: Config) -> None:
    """组员结束后，上级若已齐就回到 running。"""
    if node.parent_id is None:
        return
    parent = tree.get(node.parent_id)
    workspace = parent.active_workspace()
    if workspace is None:
        return

    message = node.return_message or "返回了，但没有写内容。"
    workspace.child_reports = [
        report for report in workspace.child_reports if report.child_id != node.id
    ]
    workspace.child_reports.append(
        ChildReport(child_id=node.id, title=node.title, message=message)
    )
    slot = _slot_for_child(workspace, node.id)
    if slot is not None:
        slot.returned = True

    if parent.open_question() is not None:
        parent.status = "running"
        return

    if isinstance(parent, Root):
        if node.node_type == "reviewer" and tree.plan_version >= cfg.repeat:
            parent.status = "finished"
            return
        nxt = _ROOT_AFTER_RETURN.get(node.node_type)
        if nxt is not None:
            workspace.status = nxt
        parent.status = "running"
        return

    if isinstance(parent, Reviewer):
        _collect_review_if_ready(tree, parent, workspace)
        return

    group = workspace.group
    if (
        parent.status == "waiting"
        and group is not None
        and group.kind == "sequence"
        and group.current_index + 1 < len(group.slots)
    ):
        next_index = group.current_index + 1
        group.current_index = next_index
        slot = group.slots[next_index]
        child = tree.get(slot.node_id)
        task_index = next_index + 1
        child.from_parent = (
            _render_assignment(parent, workspace, task_index)
            + "\n\n## 上一位成员返回\n"
            + f"{node.id} {node.title}:\n{message}"
        )
        child.position = _summarize_child_position(tree, child, cfg)
        child.return_message = ""
        child.status = "running"
        workspace.status = "dispatched"
        parent.status = "waiting"
        return

    if parent.status == "waiting" and _all_dispatched_slots_returned(tree, workspace):
        workspace.status = "review"
        parent.status = "running"


def _slot_of_type(workspace: Workspace, node_type: str) -> ChildSlot | None:
    group = workspace.group
    if group is None:
        return None
    for slot in group.slots:
        if slot.node_type == node_type:
            return slot
    return None


def _ensure_root_team(tree: Tree, root: Root, workspace: Workspace) -> None:
    if workspace.group is not None and workspace.group.slots:
        return
    slots: list[ChildSlot] = []
    for node_type in ("designer", "agent", "reviewer"):
        child = tree.add_child(root, node_type=node_type, title="", status="staged")
        slots.append(
            ChildSlot(title=child.title, node_id=child.id, node_type=node_type)
        )
    workspace.group = NodeGroup(
        kind="sequence",
        count=3,
        status="dispatched",
        slots=slots,
    )


def _doc_assignment(brief: str) -> str:
    return brief


def _build_assignment(_doc_message: str = "") -> str:
    return "你负责的范围是整个文档。"


def _review_lead_assignment(tree: Tree) -> str:
    return (
        f"人的要求：\n{tree.root_brief}\n\n"
        f"开发交回的使用方式：\n{tree.delivery}"
    )


def _review_assignment(tree: Tree, workspace: Workspace, task_index: int) -> str:
    return (
        f"人的要求：\n{tree.root_brief}\n\n"
        f"使用方式：\n{tree.delivery or '（还没有）'}\n\n"
        f"你负责第 {task_index} 部分。下面是全部分工，只核验这一部分。\n\n"
        f"{workspace.plan}"
    )


def _revise_assignment(tree: Tree, project: Path, run_id: str) -> str:
    previous = plan_doc_path(project, run_id, tree.plan_version - 1)
    current = plan_doc_path(project, run_id, tree.plan_version)
    return (
        "这是最新的改进建议。\n\n"
        f"{tree.review_note}\n\n"
        f"上一版工作手册在：{previous}\n"
        "只读，不要改那里的文件。\n\n"
        f"这一版只能写在：{current}\n"
        "写的是针对上一版的改进工作手册，不是另起一份新项目。"
    )


def _open_review_round(reviewer: Reviewer) -> None:
    active = reviewer.active_workspace()
    if active is not None and active.status == "reviewer_look" and not active.plan.strip():
        return
    for workspace in reviewer.workspaces:
        if workspace.status != "closed":
            workspace.status = "closed"
    reviewer.workspaces.append(
        Workspace(id=reviewer.next_workspace_id(), status="reviewer_look")
    )


def _open_designer_revision(designer: Designer) -> None:
    for workspace in designer.workspaces:
        if workspace.status != "closed":
            workspace.status = "closed"
    designer.workspaces.append(
        Workspace(id=designer.next_workspace_id(), status="designer_revise_look")
    )


def _open_worker_round(worker: Node) -> None:
    """下一版开发从认任务重新开始。上一版交用法的工作区关掉。"""
    for workspace in worker.workspaces:
        if workspace.status != "closed":
            workspace.status = "closed"


def _hand_child(
    tree: Tree, child: Node, text: str, project: Path, cfg: Config
) -> None:
    child.from_parent = text
    child.position = _summarize_child_position(tree, child, cfg)
    child.return_message = ""
    child.status = "running"
    key = _session_key(child)
    if has_persisted_session(project, cfg.run_id, key):
        append_context_message(
            child,
            project,
            f"## 上级交给你的新一版\n\n{text}",
            run_id=cfg.run_id,
            condense_every=cfg.condense_every,
        )


def _retire_subtree(tree: Tree, node: Node) -> None:
    """这一轮用过的人和他下面的组织留在树上，但不再开工。"""
    node.status = "finished"
    for workspace in node.workspaces:
        if workspace.status != "closed":
            workspace.status = "closed"
    for child_id in list(node.children_ids):
        _retire_subtree(tree, tree.get(child_id))


def _fresh_round(node: Node) -> bool:
    """还没干过这一角。开发没有工作区；验收负责人还停在空白的第一眼。"""
    if node.status == "finished":
        return False
    if node.node_type == "agent":
        return not node.workspaces
    if isinstance(node, Reviewer) and node.review_role == "leader":
        active = node.active_workspace()
        return (
            active is not None
            and len(node.workspaces) == 1
            and active.status == "reviewer_look"
            and not active.plan.strip()
            and not node.children_ids
        )
    return True


def _replace_root_child(tree: Tree, root: Root, slot: ChildSlot) -> Node:
    child = tree.add_child(root, node_type=slot.node_type, title="", status="staged")
    slot.node_id = child.id
    slot.title = child.title
    slot.returned = False
    return child


def _send_next(tree: Tree, node: Node, cfg: Config, project: Path) -> str:
    if not isinstance(node, Root):
        return f"{node.id} send_next 失败：只有根可以派这三个人"
    active = node.active_workspace()
    if active is None:
        return f"{node.id} send_next 失败：没有工作区"
    _ensure_root_team(tree, node, active)
    assert active.group is not None

    if active.status == "root_send_doc":
        slot = _slot_of_type(active, "designer")
        text = _doc_assignment(tree.root_brief)
        waiting = "root_wait_doc"
    elif active.status == "root_send_build":
        doc = _slot_of_type(active, "designer")
        if doc is None or not doc.returned:
            return f"{node.id} send_next 失败：设计的人还没交回"
        plan_path = plan_doc_path(project, cfg.run_id, tree.plan_version)
        if not plan_docs_ready(project, cfg.run_id, tree.plan_version):
            return f"{node.id} send_next 失败：内部计划还不在 {plan_path}"
        slot = _slot_of_type(active, "agent")
        text = _build_assignment()
        waiting = "root_wait_build"
    elif active.status == "root_send_review":
        if not tree.delivery.strip():
            return f"{node.id} send_next 失败：开发还没有交回用法"
        slot = _slot_of_type(active, "reviewer")
        text = _review_lead_assignment(tree)
        waiting = "root_wait_review"
    elif active.status == "root_send_revise":
        slot = _slot_of_type(active, "designer")
        if slot is None:
            return f"{node.id} send_next 失败：找不到写计划的人"
        tree.plan_version += 1
        text = _revise_assignment(tree, project, cfg.run_id)
        waiting = "root_wait_doc"
    else:
        return f"{node.id} send_next 失败：当前阶段是 {active.status}"

    if slot is None:
        return f"{node.id} send_next 失败：找不到要派的人"
    child = tree.get(slot.node_id)
    if active.status == "root_send_build" and child.node_type == "agent":
        if not _fresh_round(child):
            _retire_subtree(tree, child)
            child = _replace_root_child(tree, node, slot)
        else:
            _open_worker_round(child)
    elif isinstance(child, Reviewer):
        if not _fresh_round(child):
            _retire_subtree(tree, child)
            child = _replace_root_child(tree, node, slot)
        else:
            _open_review_round(child)
    elif active.status == "root_send_revise" and isinstance(child, Designer):
        _open_designer_revision(child)
    _hand_child(tree, child, text, project, cfg)
    slot.returned = False
    active.status = waiting  # type: ignore[assignment]
    node.status = "waiting"
    return f"{node.id} send_next -> {child.id} {child.title}"


def _root_continue(
    tree: Tree, node: Root, payload: dict, cfg: Config, project: Path
) -> str:
    active = node.active_workspace()
    if active is None or active.group is None:
        return f"{node.id} continue_child 失败：还没有三个人"
    child_id = str(payload.get("child_id", "")).strip()
    slot = _slot_for_child(active, child_id)
    if slot is None:
        return f"{node.id} continue_child 失败：找不到子成员 {child_id}"
    child = tree.get(slot.node_id)
    if child.status != "finished" or not slot.returned:
        return f"{node.id} continue_child 失败：子成员尚未返回"
    message = str(payload.get("message", "")).strip()
    if not message:
        return f"{node.id} continue_child 失败：message 不能为空"

    if active.status != "root_send_review" or child.node_type != "agent":
        return f"{node.id} continue_child 失败：这一步只能继续开发的人"
    child_ws = child.active_workspace()
    if child_ws is not None:
        child_ws.status = "worker_final"
    waiting = "root_wait_usage"

    append_context_message(
        child,
        project,
        f"## 上级要求继续\n\n{message}\n\n做完后再次 return_to_parent。",
        run_id=cfg.run_id,
        condense_every=cfg.condense_every,
    )
    slot.returned = False
    child.return_message = ""
    child.status = "running"
    active.status = waiting  # type: ignore[assignment]
    node.status = "waiting"
    return f"{node.id} continue_child -> {child.id}"


def apply_decision(
    tree: Tree, node: Node, decision: Decision, cfg: Config, workspace: Path
) -> str:
    """把一个回合的决定落到树上，返回一句给终端看的话。"""
    p = decision.payload
    feedback = _decision_error_feedback(node, decision)
    if feedback is not None:
        _append_decision_error(node, workspace, feedback, cfg)
        node.status = "running"
        allowed = _allowed_decisions(node) or set()
        return (
            f"{node.id} {decision.kind} 失败：当前阶段 {node.phase()} "
            f"{_allowed_text(allowed)}"
        )

    if decision.kind == "choose_work_mode":
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        if active.status != "choosing_work_mode":
            return f"{node.id} choose_work_mode 失败：当前阶段是 {active.status}"
        mode = p.get("mode")
        if mode == "self":
            active.work_mode = "self"
            active.status = "planning_self"
            node.status = "running"
            return f"{node.id} choose_work_mode(self) -> 直接执行"
        if mode == "delegate":
            active.work_mode = "delegate"
            active.status = "choosing_org_mode"
            node.status = "running"
            return f"{node.id} choose_work_mode(delegate) -> 请选择组织方式"
        return f"{node.id} choose_work_mode 失败：mode 必须是 self 或 delegate"

    if decision.kind == "choose_org_mode":
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        if active.status != "choosing_org_mode":
            return f"{node.id} choose_org_mode 失败：当前阶段是 {active.status}"
        mode = p.get("mode")
        if mode == "sequence":
            active.org_mode = "sequence"
            active.status = "designing_sequence"
            node.status = "running"
            return f"{node.id} choose_org_mode(sequence) -> 设计顺序式团队"
        if mode == "parallel":
            active.org_mode = "parallel"
            active.status = "designing_parallel"
            node.status = "running"
            return f"{node.id} choose_org_mode(parallel) -> 设计并列式团队"
        return f"{node.id} choose_org_mode 失败：mode 必须是 sequence 或 parallel"

    if decision.kind == "write_plan":
        active = node.active_workspace()
        if active is None or active.status not in {
            "planning_sequence",
            "planning_parallel",
        }:
            return f"{node.id} write_plan 失败：当前阶段是 {node.phase()}"
        if active.plan.strip():
            return f"{node.id} write_plan 失败：当前工作区 plan 已存在；大改请在 review 阶段 change_plan"
        plan = p.get("plan", "").strip()
        if not plan:
            return f"{node.id} write_plan 失败：plan 不能为空"
        active.plan = plan
        active.position = _position_summary(cfg, plan)
        active.status = "planned"
        node.status = "running"
        return f"{node.id} write_plan -> {_workspace_label(node, active)}{_summary_note(active.position)}，请创建节点组"

    if decision.kind in {
        "create_parallel_group",
        "create_sequence_group",
    }:
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        if active.status != "planned":
            return f"{node.id} 创建节点组失败：当前阶段是 {active.status}，不是 planned"
        if active.group is not None:
            return f"{node.id} 创建节点组失败：当前工作区已经有节点组，请 dispatch"

        kind = "parallel" if decision.kind == "create_parallel_group" else "sequence"
        count = int(p.get("count", 0))
        if count < 1:
            return f"{node.id} 创建节点组失败：count 必须大于 0"

        slots: list[ChildSlot] = []
        for _index in range(count):
            child = tree.add_child(
                node,
                node_type="agent",
                title="",
                status="staged",
            )
            slots.append(
                ChildSlot(title=child.title, node_id=child.id, node_type="agent")
            )
        active.group = NodeGroup(kind=kind, count=count, slots=slots)
        active.status = "group_created"
        node.status = "running"
        return f"{node.id} create_{kind}_group({count}) 成功，请调用 dispatch"

    if decision.kind == "send_review_plan":
        if not isinstance(node, Reviewer) or node.review_role != "leader":
            return f"{node.id} send_review_plan 失败：只有验收负责人可以下发"
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        if active.status != "reviewer_dispatch":
            return f"{node.id} send_review_plan 失败：当前阶段是 {node.phase()}"
        if active.group is not None:
            return f"{node.id} send_review_plan 失败：这一轮已经下发过"
        plan = str(p.get("plan", "")).strip()
        count = _review_part_count(plan)
        if count is None:
            message = (
                f"{node.id} send_review_plan 失败：每一部分单独起行，"
                "从「第 1 部分」连续编到最后一部分，中间不能跳号。"
            )
            _append_decision_error(
                node,
                workspace,
                "## 划分没有下发\n\n"
                + message
                + "\n\n请按这个格式重新调用 send_review_plan。",
                cfg,
            )
            node.status = "running"
            return message
        active.plan = plan
        active.position = _position_summary(cfg, plan)
        slots: list[ChildSlot] = []
        for _index in range(count):
            child = tree.add_child(
                node,
                node_type="reviewer",
                title="",
                status="staged",
            )
            slots.append(
                ChildSlot(title=child.title, node_id=child.id, node_type="reviewer")
            )
        active.group = NodeGroup(kind="parallel", count=count, slots=slots)
        for index, slot in enumerate(slots):
            child = tree.get(slot.node_id)
            child.return_message = ""
            child.from_parent = _review_assignment(tree, active, index + 1)
            child.position = _summarize_child_position(tree, child, cfg)
            child.status = "running"
        active.group.status = "dispatched"
        active.status = "dispatched"
        node.status = "waiting"
        return (
            f"{node.id} send_review_plan({count}) -> "
            f"{', '.join(slot.node_id for slot in slots)}"
        )

    if decision.kind == "dispatch":
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        group = active.group
        if active.status != "group_created" or group is None:
            return f"{node.id} dispatch 失败：没有待派遣的节点组"
        if group.status != "staged":
            return f"{node.id} dispatch 失败：节点组已经派遣过"
        for index, slot in enumerate(group.slots):
            child = tree.get(slot.node_id)
            task_index = index + 1
            child.return_message = ""
            runnable_now = group.kind != "sequence" or index == group.current_index
            child.from_parent = (
                _render_assignment(node, active, task_index) if runnable_now else ""
            )
            child.position = (
                _summarize_child_position(tree, child, cfg) if runnable_now else ""
            )
            child.status = "running" if runnable_now else "staged"
        group.status = "dispatched"
        active.status = "dispatched"
        node.status = "waiting"
        return f"{node.id} dispatch({group.kind}) -> {', '.join(s.node_id for s in group.slots)}"

    if decision.kind == "ask_parent":
        question_text = p.get("question", "").strip()
        if not question_text:
            return f"{node.id} ask_parent 失败：question 不能为空"
        if node.parent_id is None:
            append_context_message(
                node,
                workspace,
                (
                    "## 上级回答\n\n"
                    f"你刚才问：{question_text}\n\n"
                    "上级回答：你可以自己决定\n\n"
                    "请根据这个回答继续当前任务。"
                ),
                run_id=cfg.run_id,
                condense_every=cfg.condense_every,
            )
            node.status = "running"
            return f"{node.id} ask_parent -> 你可以自己决定"
        parent = tree.get(node.parent_id)
        active = parent.active_workspace()
        if active is None:
            return f"{node.id} ask_parent 失败：上级没有 active workspace"
        question = ChildQuestion(
            id=_next_question_id(active),
            child_id=node.id,
            question=question_text,
        )
        active.questions.append(question)
        node.status = "waiting"
        parent.status = "running"
        return f"{node.id} ask_parent -> {parent.id} {question.id}"

    if decision.kind == "answer_child_question":
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        qid = p.get("question_id", "")
        question = next((q for q in active.questions if q.id == qid), None)
        if question is None:
            return f"{node.id} answer_child_question 失败：找不到 open 问题 {qid}"
        answer = p.get("answer", "").strip()
        if not answer:
            return f"{node.id} answer_child_question 失败：answer 不能为空"
        child = tree.get(question.child_id)
        append_context_message(
            child,
            workspace,
            (
                "## 上级回答\n\n"
                f"你刚才问：{question.question}\n\n"
                f"上级回答：{answer}\n\n"
                "请根据这个回答继续当前任务。"
            ),
            run_id=cfg.run_id,
            condense_every=cfg.condense_every,
        )
        active.questions = [q for q in active.questions if q.id != qid]
        child.status = "running"
        if isinstance(node, Reviewer):
            _collect_review_if_ready(tree, node, active)
        else:
            node.status = "running" if node.open_question() else "waiting"
        return f"{node.id} answer_child_question({question.id}) -> {child.id}"

    if decision.kind == "finish_design":
        if not isinstance(node, Designer):
            return f"{node.id} finish_design 失败：只有写计划的人可以结束设计"
        active = node.active_workspace()
        if active is None or active.status not in {"designer_write", "designer_revise"}:
            return f"{node.id} finish_design 失败：当前阶段是 {node.phase()}"
        plan_path = plan_doc_path(workspace, cfg.run_id, tree.plan_version)
        if not plan_docs_ready(workspace, cfg.run_id, tree.plan_version):
            return f"{node.id} finish_design 失败：请先把计划写进 {plan_path}"
        node.return_message = "计划已写好。"
        node.status = "finished"
        _wake_parent(tree, node, cfg)
        return f"{node.id} finish_design，计划在 {plan_path}"

    if decision.kind == "send_next":
        return _send_next(tree, node, cfg, workspace)

    if decision.kind == "continue_child":
        if isinstance(node, Root):
            return _root_continue(tree, node, p, cfg, workspace)
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        if active.status != "review":
            return f"{node.id} continue_child 失败：只能在 review 阶段继续子成员"
        group = active.group
        if group is None:
            return f"{node.id} continue_child 失败：当前工作区没有子成员"
        child_id = p.get("child_id", "").strip()
        slot = _slot_for_child(active, child_id)
        if slot is None:
            return f"{node.id} continue_child 失败：找不到子成员 {child_id}"
        child = tree.get(slot.node_id)
        if child.status != "finished" or not slot.returned:
            return f"{node.id} continue_child 失败：子成员尚未返回"
        message = p.get("message", "").strip()
        if not message:
            return f"{node.id} continue_child 失败：message 不能为空"
        child_ws = child.active_workspace()
        coding = child_ws is not None and child_ws.status in {
            "summarizing_self",
            "planning_self",
        }
        follow = (
            "请根据这条进一步指令继续当前任务。做完停下来即可。"
            if coding
            else "请根据这条进一步指令继续当前任务；完成后再次 return_to_parent。"
        )
        append_context_message(
            child,
            workspace,
            f"## 上级要求继续修改\n\n{message}\n\n{follow}",
            run_id=cfg.run_id,
            condense_every=cfg.condense_every,
        )
        if coding and child_ws is not None:
            child_ws.status = "planning_self"
        slot.returned = False
        child.return_message = ""
        child.status = "running"
        active.status = "dispatched"
        node.status = "waiting"
        return f"{node.id} continue_child -> {child.id}"

    if decision.kind == "change_plan":
        active, error = _require_workspace(node)
        if error:
            return error
        assert active is not None
        if active.status != "review":
            return f"{node.id} change_plan 失败：只能在 review 阶段进入新工作区"
        if _active_group_children(tree, active):
            return f"{node.id} change_plan 失败：还有成员未返回"
        new_plan = p.get("plan", "").strip()
        failure_summary = p.get("failure_summary", "").strip()
        if not new_plan or not failure_summary:
            return f"{node.id} change_plan 失败：failure_summary 和 plan 都不能为空"
        active.status = "closed"
        fresh = Workspace(
            id=node.next_workspace_id(),
            failure_summary=failure_summary,
            plan=new_plan,
            position=_position_summary(cfg, new_plan),
            status="planned",
        )
        node.workspaces.append(fresh)
        node.status = "running"
        return f"{node.id} change_plan -> {_workspace_label(node, fresh)}{_summary_note(fresh.position)}，请重新创建节点组"

    # return_to_parent
    active = node.active_workspace()
    message = p.get("message", "")
    node.return_message = message
    # 只有开发负责人真正做完（从 review 交回）才追问用法。
    # 半路调用 return_to_parent 不能跳过开发。
    if (
        active is not None
        and active.status == "review"
        and _begin_usage_report(tree, node)
    ):
        return f"{node.id} 请交回用法，不要写代码"
    allowed_return = {
        "planning_self",
        "review",
        "member_return",
        "reviewer_filter",
        "worker_final",
    }
    if active is not None and active.status not in allowed_return:
        return f"{node.id} return_to_parent 失败：当前阶段是 {active.status}，请先完成 review"
    if active is not None and active.status in {"member_return", "reviewer_filter"}:
        normalized = _normalize_member_return(message)
        if normalized is None:
            note = (
                f"{node.id} return_to_parent 失败：正文只能使用"
                "[必须修改]、[推荐调整]、[备选优化]。"
                "没有内容的标签不要写，标签以外不要写别的话。"
            )
            _append_decision_error(
                node,
                workspace,
                "## 归类没有收下\n\n"
                + note
                + "\n\n标签单独成行，条目写在标签下面。某一类没有内容，就整段跳过。",
                cfg,
            )
            node.status = "running"
            return note
        message = normalized
        node.return_message = message
    if active is not None and active.status == "worker_final":
        tree.delivery = message.strip()
        tree.usage_for_version = tree.plan_version
    if isinstance(node, Reviewer) and node.review_role == "leader":
        tree.review_note = message.strip()
    node.status = "finished"
    _wake_parent(tree, node, cfg)
    return f"{node.id} return_to_parent，返回"


def run_tree(
    tree: Tree,
    workspace: Path,
    cfg: Config,
    *,
    dump_path: Optional[Path] = None,
    say: Callable[[str], None] = print,
    step_sink: Optional[Callable[[str], None]] = None,
) -> str:
    turns = 0
    log_lock = Lock()

    def locked_say(line: str) -> None:
        with log_lock:
            say(line)

    def locked_step_sink(line: str) -> None:
        if step_sink is None:
            return
        with log_lock:
            step_sink(line)

    while tree.root.status != "finished":
        if turns >= cfg.max_turns:
            return "回合用尽"
        ready = runnable(tree)
        if not ready:
            return "没有可跑的节点"

        remaining = cfg.max_turns - turns
        batch = select_batch(tree, ready, cfg)[:remaining]
        if not batch:
            return "没有可跑的节点"

        for offset, node in enumerate(batch):
            phase = node.phase()
            title = node.title or "root"
            shown = node.id if title == node.id else f"{node.id} {title}"
            locked_say(
                f"[{turns + offset}] {shown} · {phase}"
            )

        def run_one(node: Node) -> tuple[Node, TurnResult]:
            try:
                result = run_turn(
                    tree,
                    node,
                    workspace,
                    step_sink=locked_step_sink,
                    condense_every=cfg.condense_every,
                    max_iterations=_max_iterations_for(node, cfg),
                    run_id=cfg.run_id,
                )
                return node, result
            except Exception as exc:  # noqa: BLE001 - keep scheduler alive.
                return node, TurnResult(
                    decision=None, steps=0, note=f"异常：{exc}", error=True
                )

        results: list[tuple[int, Node, TurnResult]] = []
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {
                pool.submit(run_one, node): index for index, node in enumerate(batch)
            }
            for future in as_completed(futures):
                index = futures[future]
                node, result = future.result()
                results.append((index, node, result))

        for _, node, result in sorted(results, key=lambda item: item[0]):
            turns += 1

            if result.decision is None:
                if not result.error and _finish_look(node, result.text):
                    locked_say(f"     {node.id} 看完 -> {node.phase()}")
                elif not result.error and _finish_review_plan(node, result.text):
                    locked_say(f"     {node.id} 划分想完 -> {node.phase()}")
                elif not result.error and _finish_member_check(node, result.text):
                    locked_say(f"     {node.id} 检查完 -> {node.phase()}")
                elif not result.error and _finish_design(node):
                    locked_say(f"     {node.id} 设计完成 -> {node.phase()}")
                elif not result.error and _begin_summary(node):
                    locked_say(f"     {node.id} 停工 -> {node.phase()}")
                elif (
                    not result.error and node.phase() == "3A2_summarize"
                ):
                    locked_say(
                        "     "
                        + _finish_summary(tree, node, result.text, cfg)
                    )
                elif isinstance(node, Root):
                    locked_say(
                        "     "
                        + _continue_root_without_decision(
                            tree, node, result.text, cfg, workspace
                        )
                    )
                else:
                    _append_decision_error(
                        node,
                        workspace,
                        _missing_decision_feedback(node, result.note),
                        cfg,
                    )
                    node.status = "running"
                    locked_say(
                        f"     {node.id} 没给决定，留在原地："
                        f"{result.note or '无说明'}"
                    )
            else:
                locked_say(
                    "     "
                    + apply_decision(tree, node, result.decision, cfg, workspace)
                )

            if dump_path and turns % cfg.dump_every == 0:
                tree.dump(dump_path)

    if dump_path:
        tree.dump(dump_path)
    note = tree.review_note.strip()
    locked_say("最终验收：")
    locked_say(note if note else "（这一版没有要提交的条目）")
    return "完成"
