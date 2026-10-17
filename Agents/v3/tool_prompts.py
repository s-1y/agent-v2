"""从 prompts/tools/ 读决定工具的说明文字。"""

from __future__ import annotations

from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent / "prompts" / "tools"


def load_tool_doc(name: str) -> dict[str, str]:
    """读 prompts/tools/<name>.md。

    格式：
      ## tool
      工具总说明

      ## 字段名
      该参数说明
    """
    path = _TOOLS_DIR / f"{name}.md"
    text = path.read_text(encoding="utf-8").strip()
    sections: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(chunks).strip()
            current = line[3:].strip()
            chunks = []
        else:
            chunks.append(line)
    if current is not None:
        sections[current] = "\n".join(chunks).strip()
    if not sections:
        raise ValueError(f"{path} 没有任何 ## 分段")
    return sections
