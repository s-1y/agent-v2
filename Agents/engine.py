"""The only module that talks to OpenHands. Everyone else imports from here.

Needs the meta-agent env: pip install openhands-sdk openhands-tools.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from pydantic import Field, SecretStr

from Agents.paths import REPO_ROOT


def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs if the process has not already set them."""
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(REPO_ROOT / ".env")

from openhands.sdk import (  # noqa: E402
    Agent,
    AgentContext,
    Conversation,
    LLM,
    TextContent,
    Tool,
    ToolDefinition,
    create_mcp_tools,
    register_tool,
)
from openhands.sdk.context import KeywordTrigger, Skill, load_skills_from_dir  # noqa: E402
from openhands.sdk.context.condenser import LLMSummarizingCondenser  # noqa: E402
from openhands.sdk.event import ActionEvent, MessageEvent  # noqa: E402
from openhands.sdk.llm import Message  # noqa: E402
from openhands.sdk.tool import Action, Observation, ToolExecutor  # noqa: E402
from openhands.sdk.tool.builtins import FinishTool, ThinkTool  # noqa: E402
from openhands.tools.browser_use import BrowserToolSet  # noqa: E402
from openhands.tools.file_editor import FileEditorTool  # noqa: E402
from openhands.tools.glob import GlobTool  # noqa: E402
from openhands.tools.grep import GrepTool  # noqa: E402
from openhands.tools.task_tracker import TaskTrackerTool  # noqa: E402
from openhands.tools.terminal import TerminalTool  # noqa: E402

WorkspacePath = Union[str, Path]


def make_llm(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    usage_id: str = "agent",
) -> LLM:
    """Build an LLM from explicit args or LLM_MODEL / LLM_API_KEY / LLM_BASE_URL."""
    resolved_key = (
        api_key
        if api_key is not None
        else os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    )
    if not resolved_key:
        raise RuntimeError("LLM_API_KEY is not set")
    return LLM(
        usage_id=usage_id,
        model=model or os.getenv("LLM_MODEL", "gpt-5.5"),
        api_key=SecretStr(resolved_key),
        base_url=base_url if base_url is not None else os.getenv("LLM_BASE_URL") or None,
        reasoning_effort="low",
        # 千问默认会写很长的隐藏思考。这里只留一小段，设计阶段的思考已经在阶段提示里。
        litellm_extra_body={"enable_thinking": True, "thinking_budget": 1024},
    )


def make_condenser(
    *,
    llm: Optional[LLM] = None,
    max_size: int = 240,
    keep_first: int = 2,
    max_tokens: Optional[int] = None,
) -> LLMSummarizingCondenser:
    """Use OpenHands LLMSummarizingCondenser defaults.

    Defaults match the SDK: max_size=240, keep_first=2, max_tokens=None.
    With max_tokens=None, token pressure still uses the agent LLM's
    effective_max_input_tokens when the SDK passes it through.
    """
    return LLMSummarizingCondenser(
        llm=llm or make_llm(usage_id="condenser"),
        max_size=max_size,
        keep_first=keep_first,
        max_tokens=max_tokens,
    )


def complete_text(prompt: str, *, llm: Optional[LLM] = None, usage_id: str = "plain") -> str:
    """Ask the LLM for one plain text answer."""
    from openhands.sdk.agent.utils import make_llm_completion

    response = make_llm_completion(
        llm=llm or make_llm(usage_id=usage_id),
        messages=[Message(role="user", content=[TextContent(text=prompt)])],
    )
    if not response.message.content:
        return ""
    first = response.message.content[0]
    return first.text if isinstance(first, TextContent) else ""


def collapse_shell_script(action) -> str:
    """多行 shell 收成一条 `bash 脚本`，让 terminal 的单命令限制能跑 heredoc。"""
    command = getattr(action, "command", "") or ""
    if getattr(action, "is_input", False) or not command.strip():
        return command
    from openhands.tools.terminal.utils.command import split_bash_commands

    try:
        parts = split_bash_commands(command)
    except Exception:
        return command
    if len(parts) <= 1:
        return command
    import os
    import shlex
    import tempfile

    fd, path = tempfile.mkstemp(prefix="oh-shell-", suffix=".sh")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(command)
        if not command.endswith("\n"):
            handle.write("\n")
    os.chmod(path, 0o700)
    return f"bash {shlex.quote(path)}"


class _CollapseMultilineExecutor(ToolExecutor):
    """交给 OpenHands terminal 之前，把会被拆开的多行命令换成单条 bash。"""

    def __init__(self, inner: ToolExecutor) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def __call__(self, action, conversation=None):
        command = collapse_shell_script(action)
        if command != action.command:
            action = action.model_copy(update={"command": command})
        return self._inner(action, conversation)


class _GluedTerminalTool(TerminalTool):
    # 子类自动起名会变成别的工具名。注册和模型看见的仍是 terminal。
    name = TerminalTool.name

    @classmethod
    def create(cls, conv_state, **params):
        tools = TerminalTool.create(conv_state, **params)
        glued = []
        for tool in tools:
            glued.append(
                tool.model_copy(
                    update={"executor": _CollapseMultilineExecutor(tool.executor)}
                )
            )
        return glued


register_tool(TerminalTool.name, _GluedTerminalTool)


def _install_tool_name_glue() -> None:
    """叫错的工具名，只有参数已经能对上现有工具时才改过去。

    不补全缺失参数。理解任务阶段没有 terminal 时，exec Bash 仍然拒绝。
    """
    import re

    from openhands.sdk.agent import agent as agent_mod
    from openhands.sdk.agent import utils as agent_utils

    agent_utils.TOOL_NAME_ALIASES.update(
        {
            "exec": "terminal",
            "edit_file": "file_editor",
            "editor": "file_editor",
            "file_edit": "file_editor",
        }
    )
    arg_renames = {
        "old_string": "old_str",
        "new_string": "new_str",
        "file_path": "path",
    }
    editor_payload = {
        "command",
        "old_str",
        "new_str",
        "file_text",
        "insert_line",
        "view_range",
    }
    editor_keys = editor_payload | {"path", "security_risk", "summary"}
    terminal_keys = {"command", "is_input", "timeout", "reset", "security_risk", "summary"}
    original = agent_utils.normalize_tool_call

    def normalize_tool_call(tool_name, arguments, available_tools):
        name = tool_name
        args = dict(arguments)
        if tool_name not in available_tools:
            folded = re.sub(r"\s+", "_", tool_name.strip()).lower()
            target = agent_utils.TOOL_NAME_ALIASES.get(folded)
            if target in available_tools:
                name = folded
            for src, dst in arg_renames.items():
                if src in args and dst not in args:
                    args[dst] = args.pop(src)
        out_name, out_args = original(name, args, available_tools)
        if out_name == tool_name:
            return out_name, out_args
        if out_name == "file_editor":
            if not any(key in args for key in editor_payload):
                return tool_name, arguments
            out_args = {key: value for key, value in out_args.items() if key in editor_keys}
        elif out_name == "terminal":
            if not isinstance(out_args.get("command"), str):
                return tool_name, arguments
            out_args = {
                key: value for key, value in out_args.items() if key in terminal_keys
            }
        return out_name, out_args

    agent_utils.normalize_tool_call = normalize_tool_call
    agent_mod.normalize_tool_call = normalize_tool_call


_install_tool_name_glue()


def coding_tools() -> list[Tool]:
    """OpenHands coding hands plus the Chromium browser for 3A self-work."""
    return [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=BrowserToolSet.name),
    ]


class ReadFileAction(Action):
    path: str = Field(description="File to read, relative to the workspace or absolute.")
    start_line: int = Field(default=1, description="1-based first line to read.")
    max_lines: int = Field(default=400, description="How many lines to read at most.")


class ReadFileObservation(Observation):
    """Plain text file content."""


class _ReadFileExecutor(ToolExecutor):
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    def __call__(self, action: ReadFileAction, conversation=None) -> ReadFileObservation:
        target = Path(action.path)
        if not target.is_absolute():
            target = self._workspace / target
        if not target.is_file():
            return ReadFileObservation.from_text(f"not a file: {target}", is_error=True)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(action.start_line - 1, 0)
        chunk = lines[start : start + max(action.max_lines, 1)]
        body = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(chunk))
        tail = "" if start + len(chunk) >= len(lines) else "\n... (truncated)"
        return ReadFileObservation.from_text(f"{target}\n{body}{tail}")


class ReadFileTool(ToolDefinition[ReadFileAction, ReadFileObservation]):
    """Read a file. No writing, no shell."""

    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["ReadFileTool"]:
        ws = getattr(conv_state, "workspace", None)
        root = getattr(ws, "working_dir", None) or (str(ws) if ws else ".")
        workspace = Path(root)
        return [
            cls(
                name="read_file",
                description="Read a file from the workspace. Read-only.",
                action_type=ReadFileAction,
                observation_type=ReadFileObservation,
                executor=_ReadFileExecutor(workspace),
            )
        ]


register_tool("ReadFileTool", ReadFileTool)


def readonly_tools() -> list[Tool]:
    """Eyes only: read a file, glob, grep. No file editor, no terminal."""
    return [
        Tool(name="ReadFileTool"),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
    ]


def run_agent(
    task: str,
    workspace: WorkspacePath,
    *,
    tools: Optional[Sequence[Tool]] = None,
    skills: Optional[Sequence[Skill]] = None,
    llm: Optional[LLM] = None,
) -> Conversation:
    """Create an agent, send one task, run until it finishes, return the conversation."""
    agent = Agent(
        llm=llm or make_llm(),
        tools=list(tools) if tools is not None else coding_tools(),
        agent_context=AgentContext(skills=list(skills)) if skills else None,
    )
    conversation = Conversation(agent=agent, workspace=str(workspace))
    conversation.send_message(task)
    conversation.run()
    return conversation


def load_project_skills():
    """Load every SKILL.md under the shared Skills/ directory."""
    from Agents.paths import SKILLS_DIR

    if not SKILLS_DIR.is_dir():
        return []
    repo, knowledge, agent_skills = load_skills_from_dir(SKILLS_DIR)
    return [*repo.values(), *knowledge.values(), *agent_skills.values()]


__all__ = [
    "Action",
    "ActionEvent",
    "MessageEvent",
    "Agent",
    "AgentContext",
    "Conversation",
    "FileEditorTool",
    "FinishTool",
    "GlobTool",
    "GrepTool",
    "KeywordTrigger",
    "LLM",
    "Observation",
    "Skill",
    "TaskTrackerTool",
    "TerminalTool",
    "TextContent",
    "ThinkTool",
    "Tool",
    "ToolDefinition",
    "ToolExecutor",
    "ReadFileTool",
    "coding_tools",
    "readonly_tools",
    "create_mcp_tools",
    "load_project_skills",
    "load_skills_from_dir",
    "make_llm",
    "register_tool",
    "run_agent",
]
