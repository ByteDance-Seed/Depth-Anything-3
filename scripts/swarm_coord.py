#!/usr/bin/env python3
"""Simple multi-agent coordination CLI for this repository."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COORD_DIR = REPO_ROOT / "coordination"
STATE_FILE = COORD_DIR / "swarm_state.json"
LOCK_FILE = COORD_DIR / ".swarm_state.lock"
REPORTS_DIR = COORD_DIR / "reports"

VALID_AGENTS = {"PM", "DP3-DP", "SP", "RF", "TT"}
VALID_TASK_STATUS = {"todo", "in_progress", "blocked", "review", "done"}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_layout() -> None:
    COORD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.touch(exist_ok=True)


def default_state() -> dict[str, Any]:
    return {"locks": [], "tasks": [], "reports": []}


def normalize_repo_path(raw: str) -> str:
    p = Path(raw)
    if p.is_absolute():
        try:
            p = p.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"Path outside repo: {raw}") from exc

    normalized = Path(os.path.normpath(str(p)))
    if str(normalized).startswith(".."):
        raise ValueError(f"Path outside repo: {raw}")

    return normalized.as_posix()


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return default_state()
    with STATE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("locks", "tasks", "reports"):
        data.setdefault(key, [])
    return data


def save_state(state: dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=True)
        f.write("\n")
    tmp.replace(STATE_FILE)


def with_state(write: bool):
    class _StateContext:
        def __enter__(self):
            ensure_layout()
            self.fp = LOCK_FILE.open("r+", encoding="utf-8")
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
            self.state = load_state()
            return self.state

        def __exit__(self, exc_type, exc, tb):
            if write and exc_type is None:
                save_state(self.state)
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()

    return _StateContext()


def assert_agent(agent: str) -> None:
    if agent not in VALID_AGENTS:
        raise ValueError(f"Invalid agent '{agent}'. Valid: {sorted(VALID_AGENTS)}")


def cleanup_expired_locks(state: dict[str, Any]) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    kept = []
    for lock in state["locks"]:
        expires_at = lock.get("expires_at")
        if not expires_at:
            kept.append(lock)
            continue
        exp = dt.datetime.fromisoformat(expires_at)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=dt.timezone.utc)
        if exp > now:
            kept.append(lock)
    state["locks"] = kept


def cmd_init(_args: argparse.Namespace) -> int:
    with with_state(write=True) as state:
        if not state["tasks"]:
            state["tasks"] = [
                {
                    "id": "TASK-001",
                    "title": "Remote frame ingestion API",
                    "owner": "SP",
                    "status": "todo",
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                },
                {
                    "id": "TASK-002",
                    "title": "Streaming inference worker pipeline",
                    "owner": "SP",
                    "status": "todo",
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                },
            ]
        state.setdefault("locks", [])
        state.setdefault("reports", [])
    print(f"Initialized: {STATE_FILE}")
    return 0


def cmd_lock_acquire(args: argparse.Namespace) -> int:
    assert_agent(args.agent)
    if not args.paths:
        raise ValueError("At least one path is required")

    paths = [normalize_repo_path(p) for p in args.paths]

    with with_state(write=True) as state:
        cleanup_expired_locks(state)

        conflicts = []
        for p in paths:
            for lock in state["locks"]:
                if lock["path"] == p and lock["agent"] != args.agent:
                    conflicts.append(lock)

        if conflicts:
            print("Lock conflict detected:")
            for c in conflicts:
                print(
                    f"- {c['path']} locked by {c['agent']} "
                    f"(task={c.get('task','-')}, expires_at={c.get('expires_at','-')})"
                )
            return 2

        now = dt.datetime.now(dt.timezone.utc)
        expires_at = (now + dt.timedelta(hours=args.ttl_hours)).isoformat() if args.ttl_hours else None

        for p in paths:
            state["locks"] = [
                l for l in state["locks"] if not (l["path"] == p and l["agent"] == args.agent)
            ]
            state["locks"].append(
                {
                    "path": p,
                    "agent": args.agent,
                    "task": args.task,
                    "acquired_at": now_iso(),
                    "expires_at": expires_at,
                }
            )

    print("Lock acquired:")
    for p in paths:
        print(f"- {p}")
    return 0


def cmd_lock_release(args: argparse.Namespace) -> int:
    assert_agent(args.agent)
    paths = [normalize_repo_path(p) for p in args.paths] if args.paths else []

    with with_state(write=True) as state:
        before = len(state["locks"])
        if args.all:
            state["locks"] = [l for l in state["locks"] if l["agent"] != args.agent]
        else:
            if not paths:
                raise ValueError("Provide paths or use --all")
            path_set = set(paths)
            state["locks"] = [
                l for l in state["locks"] if not (l["agent"] == args.agent and l["path"] in path_set)
            ]
        removed = before - len(state["locks"])

    print(f"Released {removed} lock(s)")
    return 0


def cmd_lock_list(_args: argparse.Namespace) -> int:
    with with_state(write=False) as state:
        cleanup_expired_locks(state)
        locks = sorted(state["locks"], key=lambda x: (x["agent"], x["path"]))

    if not locks:
        print("No active locks")
        return 0

    print("Active locks:")
    for l in locks:
        print(
            f"- {l['path']} | agent={l['agent']} | task={l.get('task','-')} "
            f"| expires_at={l.get('expires_at','-')}"
        )
    return 0


def find_task(state: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for t in state["tasks"]:
        if t["id"] == task_id:
            return t
    return None


def cmd_task_add(args: argparse.Namespace) -> int:
    assert_agent(args.owner)
    if args.status not in VALID_TASK_STATUS:
        raise ValueError(f"Invalid status '{args.status}'")

    with with_state(write=True) as state:
        if find_task(state, args.id):
            raise ValueError(f"Task already exists: {args.id}")
        state["tasks"].append(
            {
                "id": args.id,
                "title": args.title,
                "owner": args.owner,
                "status": args.status,
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
        )

    print(f"Task added: {args.id}")
    return 0


def cmd_task_update(args: argparse.Namespace) -> int:
    if args.status and args.status not in VALID_TASK_STATUS:
        raise ValueError(f"Invalid status '{args.status}'")

    with with_state(write=True) as state:
        task = find_task(state, args.id)
        if not task:
            raise ValueError(f"Task not found: {args.id}")

        if args.owner:
            assert_agent(args.owner)
            task["owner"] = args.owner
        if args.status:
            task["status"] = args.status
        if args.title:
            task["title"] = args.title
        task["updated_at"] = now_iso()

    print(f"Task updated: {args.id}")
    return 0


def cmd_task_list(args: argparse.Namespace) -> int:
    with with_state(write=False) as state:
        tasks = state["tasks"]

    if args.status:
        tasks = [t for t in tasks if t["status"] == args.status]

    if not tasks:
        print("No tasks")
        return 0

    tasks = sorted(tasks, key=lambda x: x["id"])
    for t in tasks:
        print(
            f"- {t['id']} | {t['status']} | owner={t['owner']} | {t['title']} "
            f"| updated_at={t.get('updated_at','-')}"
        )
    return 0


def write_report_markdown(entry: dict[str, Any]) -> Path:
    ts = dt.datetime.fromisoformat(entry["timestamp"]).astimezone(dt.timezone.utc)
    day_dir = REPORTS_DIR / ts.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    task = entry.get("task", "NO-TASK")
    safe_task = "".join(ch for ch in task if ch.isalnum() or ch in ("-", "_")) or "NO-TASK"
    filename = f"{ts.strftime('%H%M%S')}_{entry['agent']}_{safe_task}.md"
    path = day_dir / filename

    lines = [
        f"# Session Report",
        "",
        f"- Timestamp: {entry['timestamp']}",
        f"- Agent: {entry['agent']}",
        f"- Task: {entry.get('task', '-')}",
        f"- Summary: {entry['summary']}",
        f"- Files: {', '.join(entry.get('files', [])) if entry.get('files') else '-'}",
        f"- Next: {entry.get('next', '-')}",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_files_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [normalize_repo_path(p.strip()) for p in raw.split(",") if p.strip()]


def cmd_report_add(args: argparse.Namespace) -> int:
    assert_agent(args.agent)
    files = parse_files_csv(args.files)
    entry = {
        "timestamp": now_iso(),
        "agent": args.agent,
        "task": args.task,
        "summary": args.summary,
        "files": files,
        "next": args.next,
    }

    with with_state(write=True) as state:
        state["reports"].append(entry)

    md_path = write_report_markdown(entry)
    print(f"Report added: {md_path.relative_to(REPO_ROOT)}")
    return 0


def cmd_report_list(args: argparse.Namespace) -> int:
    with with_state(write=False) as state:
        reports = list(state["reports"])

    reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    if args.limit:
        reports = reports[: args.limit]

    if not reports:
        print("No reports")
        return 0

    for r in reports:
        print(
            f"- {r.get('timestamp','-')} | agent={r.get('agent','-')} | "
            f"task={r.get('task','-')} | {r.get('summary','')}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Swarm coordination helper")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize coordination state")
    p_init.set_defaults(func=cmd_init)

    p_lock = sub.add_parser("lock", help="Lock operations")
    lock_sub = p_lock.add_subparsers(dest="lock_cmd", required=True)

    p_acq = lock_sub.add_parser("acquire", help="Acquire file locks")
    p_acq.add_argument("--agent", required=True)
    p_acq.add_argument("--task", default="")
    p_acq.add_argument("--ttl-hours", type=float, default=8.0)
    p_acq.add_argument("paths", nargs="+")
    p_acq.set_defaults(func=cmd_lock_acquire)

    p_rel = lock_sub.add_parser("release", help="Release file locks")
    p_rel.add_argument("--agent", required=True)
    p_rel.add_argument("--all", action="store_true", help="Release all locks for agent")
    p_rel.add_argument("paths", nargs="*")
    p_rel.set_defaults(func=cmd_lock_release)

    p_list = lock_sub.add_parser("list", help="List active locks")
    p_list.set_defaults(func=cmd_lock_list)

    p_task = sub.add_parser("task", help="Task operations")
    task_sub = p_task.add_subparsers(dest="task_cmd", required=True)

    p_tadd = task_sub.add_parser("add", help="Add a task")
    p_tadd.add_argument("--id", required=True)
    p_tadd.add_argument("--title", required=True)
    p_tadd.add_argument("--owner", required=True)
    p_tadd.add_argument("--status", default="todo")
    p_tadd.set_defaults(func=cmd_task_add)

    p_tupd = task_sub.add_parser("update", help="Update a task")
    p_tupd.add_argument("--id", required=True)
    p_tupd.add_argument("--status")
    p_tupd.add_argument("--owner")
    p_tupd.add_argument("--title")
    p_tupd.set_defaults(func=cmd_task_update)

    p_tlist = task_sub.add_parser("list", help="List tasks")
    p_tlist.add_argument("--status", choices=sorted(VALID_TASK_STATUS))
    p_tlist.set_defaults(func=cmd_task_list)

    p_report = sub.add_parser("report", help="Report operations")
    report_sub = p_report.add_subparsers(dest="report_cmd", required=True)

    p_radd = report_sub.add_parser("add", help="Add session report")
    p_radd.add_argument("--agent", required=True)
    p_radd.add_argument("--task", default="")
    p_radd.add_argument("--summary", required=True)
    p_radd.add_argument("--files", default="")
    p_radd.add_argument("--next", default="")
    p_radd.set_defaults(func=cmd_report_add)

    p_rlist = report_sub.add_parser("list", help="List session reports")
    p_rlist.add_argument("--limit", type=int, default=10)
    p_rlist.set_defaults(func=cmd_report_list)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
