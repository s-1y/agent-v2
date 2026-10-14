"""V2 runtime knobs that are meant to be tuned by humans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_MAX_TURNS = 40
DEFAULT_DUMP_EVERY = 1
# OpenHands LLMSummarizingCondenser default max_size.
DEFAULT_CONDENSE_EVERY = 240
DEFAULT_MAX_PARALLEL_PLAN = 4
DEFAULT_MAX_PARALLEL_SELF_WORK = 1
DEFAULT_MAX_SELF_ITERATIONS = 120
POSITION_SUMMARY_MAX_CHARS = 40
CHAIN_POSITION_MAX_CHARS = 180


@dataclass
class Config:
    max_turns: int = DEFAULT_MAX_TURNS
    dump_every: int = DEFAULT_DUMP_EVERY
    condense_every: int = DEFAULT_CONDENSE_EVERY
    max_parallel_plan: int = DEFAULT_MAX_PARALLEL_PLAN
    max_parallel_self_work: int = DEFAULT_MAX_PARALLEL_SELF_WORK
    # Single self-work SDK iteration cap per run. Set to 0 to disable.
    max_self_iterations: int = DEFAULT_MAX_SELF_ITERATIONS
    # 同一次运行的节点会话目录名。同一工作区里另一次运行必须用另一个 id。
    run_id: str = ""
    position_summarizer: Optional[Callable[[str], str]] = None
    chain_summarizer: Optional[Callable[[str], str]] = None
