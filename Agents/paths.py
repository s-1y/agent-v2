"""Single place for repo / Agents / shared directories. No sys.path edits here."""

from __future__ import annotations

import os
from pathlib import Path

AGENTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AGENTS_ROOT.parent
SKILLS_DIR = REPO_ROOT / "Skills"
TOOLS_DIR = REPO_ROOT / "Tools"
LOGGING_DIR = REPO_ROOT / "logging"
WORKSPACE_DIR = REPO_ROOT / "workspace"
BENCH_DIR = REPO_ROOT / "bench" / "mini_pretrain"

DIAG_BENCH_ENV = os.environ.get("DIAG_BENCH_ENV", "diag-bench")
