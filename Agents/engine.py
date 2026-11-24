"""The only module that talks to OpenHands. Everyone else imports from here.

Needs the meta-agent env: pip install openhands-sdk openhands-tools.
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from collections.abc import Callable, Sequence
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


def _uses_qwen_thinking(model: str, base_url: Optional[str]) -> bool:
    """百炼千问用 thinking_budget。它和 OpenAI 的 reasoning_effort 不能同时出现。"""
    name = model.lower()
    url = (base_url or "").lower()
    return "qwen" in name or "dashscope" in url or "aliyuncs" in url


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
    resolved_model = model or os.getenv("LLM_MODEL", "gpt-5.5")
    resolved_base = base_url if base_url is not None else os.getenv("LLM_BASE_URL") or None
    # SDK 默认会带 reasoning_effort。千问兼容接口再带 thinking_budget 时会被拒绝。
    if _uses_qwen_thinking(resolved_model, resolved_base):
        reasoning_effort = None
        extra_body = {"enable_thinking": True, "thinking_budget": 1024}
    else:
        reasoning_effort = "low"
        extra_body = {}
    return LLM(
        usage_id=usage_id,
        model=resolved_model,
        api_key=SecretStr(resolved_key),
        base_url=resolved_base,
        reasoning_effort=reasoning_effort,
        litellm_extra_body=extra_body,
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


_META_KEYS = {"security_risk", "summary"}
_SCHEMA_CACHE: dict[str, tuple[set[str], set[str]] | None] = {}
_CLASS_BY_NAME: dict | None = None


def _action_model(cls):
    """从工具类的泛型参数里取出参数模型，不真正创建工具。"""
    from openhands.sdk.tool.tool import Action

    if cls is None:
        return None
    for klass in getattr(cls, "__mro__", ()):
        meta = getattr(klass, "__pydantic_generic_metadata__", None) or {}
        args = meta.get("args") or ()
        candidate = args[0] if args else None
        if isinstance(candidate, type) and issubclass(candidate, Action):
            return candidate
    return None


def _tool_classes_by_runtime_name() -> dict:
    import importlib
    import inspect

    from openhands.sdk.tool.registry import get_tool_module_qualnames
    from openhands.sdk.tool.tool import ToolDefinition

    global _CLASS_BY_NAME
    if _CLASS_BY_NAME is not None:
        return _CLASS_BY_NAME
    found = {}
    for module_name in set(get_tool_module_qualnames().values()):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, ToolDefinition) or cls is ToolDefinition:
                continue
            runtime_name = getattr(cls, "name", "")
            if runtime_name and runtime_name not in found:
                found[runtime_name] = cls
    _CLASS_BY_NAME = found
    return found


def _tool_argument_schema(name: str) -> tuple[set[str], set[str]] | None:
    """这个工具必填、可选的参数名。看不出形状就不当成可识别。"""
    if name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[name]
    schema: tuple[set[str], set[str]] | None = None
    try:
        action = _action_model(_tool_classes_by_runtime_name().get(name))
        if action is not None:
            required: set[str] = set()
            optional: set[str] = set()
            for field_name, field_info in action.model_fields.items():
                if field_name in _META_KEYS:
                    continue
                if field_info.is_required():
                    required.add(field_name)
                else:
                    optional.add(field_name)
            schema = (required, optional)
    except Exception:
        schema = None
    _SCHEMA_CACHE[name] = schema
    return schema


def _unique_tool_for_arguments(arguments: dict, available_tools) -> str | None:
    """参数形状只可能属于一个现有工具时，返回那个注册名。

    必填项盖得最严的那个才算。空参数对得上好几种“做完了”的工具，不猜。
    """
    provided = {key for key in arguments if key not in _META_KEYS}
    if not provided:
        return None
    best: list[tuple[int, str]] = []
    for name in available_tools:
        schema = _tool_argument_schema(str(name))
        if schema is None:
            continue
        required, optional = schema
        if not required <= provided or not provided <= (required | optional):
            continue
        best.append((len(required), str(name)))
    if not best:
        return None
    best.sort(reverse=True)
    if len(best) == 1 or best[0][0] > best[1][0]:
        return best[0][1]
    return None


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
        if out_name == tool_name and out_name not in available_tools:
            matched = _unique_tool_for_arguments(args, available_tools)
            if matched:
                out_name, out_args = original(matched, args, available_tools)
            else:
                shown = "、".join(sorted(str(item) for item in available_tools))
                raise ValueError(
                    f"你在调用工具，但 `{tool_name}` 不是可用的名字。"
                    f"能用的是：{shown}"
                )
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


class _BrowserLease:
    """一次只有一个会话占用共享浏览器，离开时关掉页面。"""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._owner: Optional[str] = None

    def key(self, conversation) -> str:
        if conversation is None:
            return "anon"
        return str(getattr(conversation, "id", id(conversation)))

    def acquire(self, conversation) -> None:
        key = self.key(conversation)
        with self._cv:
            while self._owner not in (None, key):
                self._cv.wait()
            self._owner = key

    def release(self, conversation, close: Callable[[], None]) -> None:
        key = self.key(conversation)
        with self._cv:
            if self._owner != key:
                return
        try:
            close()
        finally:
            with self._cv:
                if self._owner == key:
                    self._owner = None
                    self._cv.notify_all()


_BROWSER_LEASE = _BrowserLease()


def _close_shared_browser() -> None:
    """关掉当前页面，并换一个空的浏览器配置，下一个人打开的是自己的。"""
    executor = BrowserToolSet._shared_executor
    if executor is None or not getattr(executor, "_initialized", False):
        return
    try:
        executor._async_executor.run_async(executor.close_browser, timeout=30)
    except Exception:
        return
    old = executor._config.get("user_data_dir")
    executor._config["user_data_dir"] = str(
        Path.home() / ".config" / "browseruse" / "profiles" / uuid.uuid4().hex
    )
    if old and "browseruse/profiles/" in str(old):
        shutil.rmtree(old, ignore_errors=True)


def release_browser(conversation) -> None:
    """这个会话的这一轮结束了。若它占着浏览器，关掉再把队列放行。"""
    _BROWSER_LEASE.release(conversation, _close_shared_browser)


def _is_shell_launcher(path: Path) -> bool:
    """脚本和 snap 包装都会自己退出，browser-use 会以为浏览器没起来。"""
    if "/snap/" in path.as_posix() or path.name == "chromium-browser":
        return True
    try:
        with path.open("rb") as handle:
            head = handle.read(2)
    except OSError:
        return True
    return head == b"#!"


def _playwright_chromium() -> Path | None:
    """本机 Playwright 缓存里的 Chrome。版本号取最大的一份。"""
    caches = [
        Path.home() / ".cache" / "ms-playwright",
        Path.home() / "Library" / "Caches" / "ms-playwright",
    ]
    relatives = (
        "chrome-linux64/chrome",
        "chrome-linux/chrome",
        "chrome-win64/chrome.exe",
        "chrome-win/chrome.exe",
        "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "chrome-mac/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    )
    found: list[tuple[int, Path]] = []
    for cache in caches:
        if not cache.is_dir():
            continue
        for directory in cache.glob("chromium-*"):
            suffix = directory.name.removeprefix("chromium-")
            version = int(suffix) if suffix.isdigit() else 0
            for relative in relatives:
                candidate = directory / relative
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    found.append((version, candidate))
    if not found:
        return None
    found.sort()
    return found[-1][1]


def _install_browser_binary() -> None:
    """用 Playwright 的 Chrome，并关掉沙箱。系统的 chromium-browser 是 snap 脚本，进程会在调试端口打开前退出。"""
    from openhands.tools.browser_use.impl import BrowserToolExecutor

    if getattr(BrowserToolExecutor, "_meta_agent_chromium", False):
        return
    original = BrowserToolExecutor.check_chromium_available
    playwright = _playwright_chromium()

    def check_chromium_available() -> str | None:
        if playwright is not None and playwright.is_file():
            return str(playwright)
        found = original()
        if found and _is_shell_launcher(Path(found)):
            return None
        return found

    original_init = BrowserToolExecutor.__init__

    def __init__(self, *args, **kwargs):
        # 这台系统关了非特权用户命名空间。Playwright 的 Chrome 不带 --no-sandbox 会立刻退出。
        kwargs["chromium_sandbox"] = False
        if playwright is not None:
            kwargs["executable_path"] = str(playwright)
        original_init(self, *args, **kwargs)

    BrowserToolExecutor.check_chromium_available = staticmethod(check_chromium_available)  # type: ignore[method-assign]
    BrowserToolExecutor.__init__ = __init__  # type: ignore[method-assign]
    BrowserToolExecutor._meta_agent_chromium = True


def _install_browser_lease() -> None:
    from openhands.tools.browser_use.impl import BrowserToolExecutor

    if getattr(BrowserToolExecutor, "_meta_agent_lease", False):
        return
    original = BrowserToolExecutor.__call__

    def __call__(self, action, conversation=None):
        _BROWSER_LEASE.acquire(conversation)
        return original(self, action, conversation)

    BrowserToolExecutor.__call__ = __call__
    BrowserToolExecutor._meta_agent_lease = True


_install_browser_binary()
_install_browser_lease()


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
    """计划会话全程可用：读文件、搜索，以及 Chromium。

    没有编辑器和终端，所以这份会话改不了项目文件。
    浏览器从第一次开工就在，后面的审阅阶段也还在。
    """
    return [
        Tool(name="ReadFileTool"),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=BrowserToolSet.name),
    ]


def result_view_tools() -> list[Tool]:
    """验收只用终端和浏览器，按用法去操作。没有读文件、搜索和编辑器。"""
    return [
        Tool(name=TerminalTool.name),
        Tool(name=BrowserToolSet.name),
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
    "result_view_tools",
    "create_mcp_tools",
    "load_project_skills",
    "load_skills_from_dir",
    "make_llm",
    "register_tool",
    "run_agent",
]
