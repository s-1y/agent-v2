"""主循环：挑一个可跑的节点，跑一个回合，按决定推进 workspace 状态机。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

from Agents.v2.config import Config
from Agents.v2.context import _render_chain_raw, render_chain_source
from Agents.v2.runtime import TurnResult, append_context_message, run_turn
from Agents.v2.tools import Decision
from Agents.v2.tree import (
    ChildQuestion,
    ChildReport,
    ChildSlot,
    Node,
    NodeGroup,
    Tree,
    Workspace,
    WorkspaceStatus,
)


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
        int(node.id[1:]),
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
        return None, f"{node.id} 当前没有工作区，请先 write_plan"
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
        "1_orient": {"finish_orient"},
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
    node.status = "finished"
    if node.parent_id is not None:
        _wake_parent(tree, node, cfg)
        return f"{node.id} 总结完成，返回"
    return f"{node.id} 总结完成"


def _finish_design(node: Node) -> bool:
    """设计步不交决定即视为答完，推进到写 plan。"""
    workspace = node.active_workspace()
    if workspace is None or workspace.status not in _DESIGN_NEXT_STATUS:
        return False
    workspace.status = _DESIGN_NEXT_STATUS[workspace.status]
    node.status = "running"
    return True


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
        slot.title = f"子任务 {task_index}"
        child.title = slot.title
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

    if decision.kind == "finish_orient":
        active = node.active_workspace()
        if active is None:
            active = Workspace(
                id=node.next_workspace_id(),
                status="choosing_work_mode",
            )
            node.workspaces.append(active)
        elif active.status == "planning":
            active.status = "choosing_work_mode"
        elif active.status != "choosing_work_mode":
            return f"{node.id} finish_orient 失败：当前阶段是 {active.status}"
        node.status = "running"
        return f"{node.id} finish_orient -> 请选择自己干还是分工"

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
        if active is None:
            active = Workspace(id=node.next_workspace_id())
            node.workspaces.append(active)
        if active.status not in {
            "planning",
            "planning_sequence",
            "planning_parallel",
        } or active.plan.strip():
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
        node_type = "agent"
        for index in range(count):
            title = f"{kind}-{index + 1}"
            child = tree.add_child(
                node,
                node_type=node_type,
                title=title,
                status="staged",
            )
            slots.append(ChildSlot(title=title, node_id=child.id, node_type=node_type))
        active.group = NodeGroup(kind=kind, count=count, slots=slots)
        active.status = "group_created"
        node.status = "running"
        return f"{node.id} create_{kind}_group({count}) 成功，请调用 dispatch"

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
            slot.title = f"子任务 {task_index}"
            child.title = slot.title
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
        if node.parent_id is None:
            return f"{node.id} ask_parent 失败：根节点没有上级"
        parent = tree.get(node.parent_id)
        active = parent.active_workspace()
        if active is None:
            return f"{node.id} ask_parent 失败：上级没有 active workspace"
        question = ChildQuestion(
            id=_next_question_id(active),
            child_id=node.id,
            question=p.get("question", "").strip(),
        )
        if not question.question:
            return f"{node.id} ask_parent 失败：question 不能为空"
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
        node.status = "running" if node.open_question() else "waiting"
        return f"{node.id} answer_child_question({question.id}) -> {child.id}"

    if decision.kind == "continue_child":
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
    if active is not None and active.status not in {
        "planning_self",
        "review",
    }:
        return f"{node.id} return_to_parent 失败：当前阶段是 {active.status}，请先完成 review"
    node.return_message = p.get("message", "")
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
            locked_say(
                f"[{turns + offset}] {node.id} {title} · {phase}"
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
                if not result.error and _finish_design(node):
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
                elif node.parent_id is None:
                    locked_say(f"     {node.id} 没给决定：{result.note or '无说明'}")
                    return "根节点未给决定"
                else:
                    node.return_message = (
                        node.return_message
                        or result.note
                        or "这一回合没有给出决定就停了。"
                    )
                    node.status = "finished"
                    _wake_parent(tree, node, cfg)
                    locked_say(f"     {node.id} 没给决定，强制结束")
            else:
                locked_say(
                    "     "
                    + apply_decision(tree, node, result.decision, cfg, workspace)
                )

            if dump_path and turns % cfg.dump_every == 0:
                tree.dump(dump_path)

    if dump_path:
        tree.dump(dump_path)
    return "完成"
