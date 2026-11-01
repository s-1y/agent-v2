"""入口：给一句要求，从空工作区把项目做起来。

    python -m Agents.v3.run --brief "..." [--workspace DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from Agents.engine import complete_text
from Agents.paths import REPO_ROOT
from Agents.run_log import quiet_third_party_logs
from Agents.v3.config import (
    DEFAULT_CONDENSE_EVERY,
    DEFAULT_DUMP_EVERY,
    DEFAULT_MAX_PARALLEL_PLAN,
    DEFAULT_MAX_PARALLEL_SELF_WORK,
    DEFAULT_MAX_SELF_ITERATIONS,
    DEFAULT_MAX_TURNS,
    DEFAULT_REPEAT,
    CHAIN_POSITION_MAX_CHARS,
    POSITION_SUMMARY_MAX_CHARS,
    Config,
)
from Agents.v3.scheduler import run_tree
from Agents.v3.tree import Tree


def _position_summarizer(plan: str) -> str:
    prompt = f"""请把下面这份 plan 压缩成一个很短的树中位置路标。

要求：
- 只输出一行中文。
- 不要解释，不要编号，不要 Markdown。
- 最多 {POSITION_SUMMARY_MAX_CHARS} 个汉字。
- 说明这个工作区主要在做什么。

PLAN:
{plan}
"""
    return complete_text(prompt, usage_id="position").strip().splitlines()[0]


def _chain_summarizer(chain: str) -> str:
    prompt = f"""请根据下面这一串任务分工链，压缩成给子任务成员看的位置说明。

要求：
- 只输出一段中文，不要编号，不要 Markdown。
- 最多 {CHAIN_POSITION_MAX_CHARS} 个汉字。
- 重点说明整体项目在做什么，以及当前成员负责的这块大概属于什么功能。
- 不要详细复述每一层拆解，不要展开完整 plan。

分工链：
{chain}
"""
    return complete_text(prompt, usage_id="chain-position").strip().splitlines()[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent V3")
    parser.add_argument("--brief", help="给根节点的那句要求。")
    parser.add_argument("--brief-file", help="从文件里读要求。")
    parser.add_argument("--workspace", help="工作区目录，默认 workspace/v3/<run_id>。")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help="做完这么多版就停。1 是只做第一版；3 是第一版再改进两次。最后一版的验收打在命令行上。",
    )
    parser.add_argument("--dump-every", type=int, default=DEFAULT_DUMP_EVERY)
    parser.add_argument(
        "--condense-every",
        type=int,
        default=DEFAULT_CONDENSE_EVERY,
        help="传给 OpenHands condenser 的 max_size；默认 240（SDK 默认）。0 关掉压缩。",
    )
    parser.add_argument(
        "--max-parallel-plan", type=int, default=DEFAULT_MAX_PARALLEL_PLAN
    )
    parser.add_argument(
        "--max-parallel-self-work", type=int, default=DEFAULT_MAX_PARALLEL_SELF_WORK
    )
    parser.add_argument(
        "--max-self-iterations",
        type=int,
        default=DEFAULT_MAX_SELF_ITERATIONS,
        help="单个 self-work 阶段每次开工的 SDK iteration 上限；0 是不限制。",
    )
    parser.add_argument(
        "--viz", action="store_true", help="同时启动网页，可视化当前工作树。"
    )
    parser.add_argument("--viz-host", default="127.0.0.1")
    parser.add_argument("--viz-port", type=int, default=8765)
    parser.add_argument(
        "--resume-run",
        help="接上 logging/v3/<run_id>.tree.json 继续。必须带当时的 --workspace。回合从 0 另计。",
    )
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat 至少是 1")

    log_dir = REPO_ROOT / "logging" / "v3"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_run:
        if not args.workspace:
            parser.error("接上旧运行时必须带当时的 --workspace")
        run_id = args.resume_run
        dump_path = log_dir / f"{run_id}.tree.json"
        if not dump_path.is_file():
            parser.error(f"找不到 {dump_path}")
        tree = Tree.load(dump_path)
        brief = tree.root_brief
        workspace = Path(args.workspace)
        if not workspace.is_dir():
            parser.error(f"工作区不存在：{workspace}")
    else:
        if args.brief_file:
            brief = Path(args.brief_file).read_text(encoding="utf-8").strip()
        elif args.brief:
            brief = args.brief
        else:
            parser.error("需要 --brief 或 --brief-file")
        run_id = time.strftime("%Y%m%d-%H%M%S")
        workspace = (
            Path(args.workspace)
            if args.workspace
            else REPO_ROOT / "workspace" / "v3" / run_id
        )
        workspace.mkdir(parents=True, exist_ok=True)
        dump_path = log_dir / f"{run_id}.tree.json"
        tree = Tree()
        tree.create_root(brief)
    log_path = log_dir / f"{run_id}.log"

    cfg = Config(
        max_turns=args.max_turns,
        repeat=args.repeat,
        dump_every=args.dump_every,
        condense_every=args.condense_every,
        max_parallel_plan=args.max_parallel_plan,
        max_parallel_self_work=args.max_parallel_self_work,
        max_self_iterations=args.max_self_iterations,
        position_summarizer=_position_summarizer,
        chain_summarizer=_chain_summarizer,
        run_id=run_id,
    )

    handle = log_path.open("a", encoding="utf-8")
    steps_handle = (log_dir / f"{run_id}.steps.jsonl").open("a", encoding="utf-8")

    def say(line: str) -> None:
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    def step_sink(line: str) -> None:
        steps_handle.write(line + "\n")
        steps_handle.flush()

    quiet_third_party_logs()
    say(f"工作区 {workspace}")
    say(f"运行   {run_id}")
    say(f"要求   {brief}")
    say(f"轮数   {args.repeat}（第 1 版设计，之后改进 {args.repeat - 1} 次）")
    if args.resume_run:
        say(f"接上   根在 {tree.get(tree.root_id).phase()}，计划第 {tree.plan_version} 版")
    if args.viz:
        from Agents.v3.viz.server import start_viz_server

        tree.dump(dump_path)
        start_viz_server(dump_path, cfg, host=args.viz_host, port=args.viz_port)
        say(f"可视化 http://{args.viz_host}:{args.viz_port}/")
    reason: str | None = None
    try:
        reason = run_tree(
            tree, workspace, cfg, dump_path=dump_path, say=say, step_sink=step_sink
        )
    finally:
        tree.dump(dump_path)
        if reason is not None:
            say(f"结束：{reason}")
            say(f"树   {dump_path}")
            say(f"日志 {log_path}")
        handle.close()
        steps_handle.close()

    return 0 if reason == "完成" else 1


if __name__ == "__main__":
    sys.exit(main())
