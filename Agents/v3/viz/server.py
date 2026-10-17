"""只读可视化服务：读 run 落盘的 tree.json，给网页提供树拓扑和节点详情。

    python -m Agents.v3.viz.server --tree logging/v3/<run_id>.tree.json [--port 8765]

working 节点由 select_batch(tree, runnable(tree), cfg) 推算：主循环每回合落盘，
下一批开始前磁盘上的树就是本批开始时的状态。限制：
- --dump-every > 1 时推算和 phase 都会滞后。
- 独立运行时 --max-parallel-* 要和那次 run 一致，否则 working 可能不准。
"""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from Agents.v3.config import (
    DEFAULT_MAX_PARALLEL_PLAN,
    DEFAULT_MAX_PARALLEL_SELF_WORK,
    Config,
)
from Agents.v3.scheduler import runnable, select_batch
from Agents.v3.tree import Node, Tree

_INDEX_HTML = Path(__file__).resolve().parent / "index.html"
_FINISH_MARK = "结束："


class TreeSource:
    """按文件修改时间缓存 tree.json；读到半个文件时沿用上一次成功的结果。"""

    def __init__(self, dump_path: Path, cfg: Config) -> None:
        self.dump_path = Path(dump_path)
        self.log_path = self.dump_path.with_name(
            self.dump_path.name.removesuffix(".tree.json") + ".log"
        )
        self.cfg = cfg
        self._lock = threading.Lock()
        self._tree: Optional[Tree] = None
        self._mtime: float = 0.0

    def load(self) -> tuple[Optional[Tree], float]:
        with self._lock:
            try:
                mtime = self.dump_path.stat().st_mtime
            except FileNotFoundError:
                return self._tree, self._mtime
            if self._tree is None or mtime != self._mtime:
                try:
                    self._tree = Tree.load(self.dump_path)
                    self._mtime = mtime
                except (ValueError, OSError):
                    pass
            return self._tree, self._mtime

    def run_ended(self, tree: Tree) -> bool:
        if tree.root.status == "finished":
            return True
        try:
            return _FINISH_MARK in self.log_path.read_text(encoding="utf-8")
        except OSError:
            return False

    def working_ids(self, tree: Tree) -> set[str]:
        if self.run_ended(tree):
            return set()
        return {node.id for node in select_batch(tree, runnable(tree), self.cfg)}


def _node_summary(node: Node, working: set[str]) -> dict[str, Any]:
    workspace = node.active_workspace()
    return {
        "id": node.id,
        "parent_id": node.parent_id,
        "title": node.title,
        "status": node.status,
        "phase": node.phase(),
        "children_ids": node.children_ids,
        "workspace": (
            {
                "id": workspace.id,
                "status": workspace.status,
                "work_mode": workspace.work_mode,
                "org_mode": workspace.org_mode,
            }
            if workspace is not None
            else None
        ),
        "working": node.id in working,
    }


def tree_payload(source: TreeSource) -> dict[str, Any]:
    tree, mtime = source.load()
    if tree is None:
        return {"ready": False, "message": f"等待首次落盘：{source.dump_path}"}
    working = source.working_ids(tree)
    return {
        "ready": True,
        "root_id": tree.root_id,
        "root_brief": tree.root_brief,
        "finished": source.run_ended(tree),
        "updated_at": mtime,
        "nodes": {
            node_id: _node_summary(node, working)
            for node_id, node in tree.nodes.items()
        },
    }


def node_payload(source: TreeSource, node_id: str) -> Optional[dict[str, Any]]:
    tree, _ = source.load()
    if tree is None or node_id not in tree.nodes:
        return None
    node = tree.get(node_id)
    return {
        "id": node.id,
        "title": node.title,
        "status": node.status,
        "phase": node.phase(),
        "from_parent": tree.from_parent(node),
        "return_message": node.return_message,
        "workspaces": [
            {
                "id": ws.id,
                "status": ws.status,
                "work_mode": ws.work_mode,
                "org_mode": ws.org_mode,
                "failure_summary": ws.failure_summary,
                "plan": ws.plan,
                "questions": [q.model_dump() for q in ws.questions],
                "child_reports": [r.model_dump() for r in ws.child_reports],
            }
            for ws in node.workspaces
        ],
    }


def _make_handler(source: TreeSource) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                self._send(
                    HTTPStatus.OK, _INDEX_HTML.read_bytes(), "text/html; charset=utf-8"
                )
            elif path == "/api/tree":
                self._json(tree_payload(source))
            elif path.startswith("/api/node/"):
                payload = node_payload(source, path.removeprefix("/api/node/"))
                if payload is None:
                    self._json({"error": "node not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._json(payload)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def make_server(
    dump_path: Path, cfg: Config, host: str, port: int
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(TreeSource(dump_path, cfg)))
    server.daemon_threads = True
    return server


def start_viz_server(
    dump_path: Path, cfg: Config, host: str = "127.0.0.1", port: int = 8765
) -> threading.Thread:
    server = make_server(dump_path, cfg, host, port)
    thread = threading.Thread(target=server.serve_forever, name="v3-viz", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent V3 工作树可视化")
    parser.add_argument("--tree", required=True, help="run 落盘的 <run_id>.tree.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--max-parallel-plan", type=int, default=DEFAULT_MAX_PARALLEL_PLAN
    )
    parser.add_argument(
        "--max-parallel-self-work", type=int, default=DEFAULT_MAX_PARALLEL_SELF_WORK
    )
    args = parser.parse_args(argv)

    cfg = Config(
        max_parallel_plan=args.max_parallel_plan,
        max_parallel_self_work=args.max_parallel_self_work,
    )
    server = make_server(Path(args.tree), cfg, args.host, args.port)
    print(f"可视化 http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
