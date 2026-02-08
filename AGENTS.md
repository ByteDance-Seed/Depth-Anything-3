# AGENTS.md

This document defines how multi-session Codex collaborators work in this repository.

## Mission
Build a Visual SLAM-oriented system on top of Depth Anything 3, with:
- edge capture on macOS / robot
- remote GPU inference and mapping
- outputs usable for navigation

## Agent Roles
- `PM` (this session): scope, priorities, task slicing, integration gate
- `DP3-DP` (Design Programmer): UI/UX dashboards, visualization, telemetry views
- `SP` (System Programmer): backend APIs, ROS2 bridge, dataflow, performance
- `RF` (Refactor): cleanup, modularization, interfaces, technical debt control
- `TT` (Tester): test plans, regression checks, latency and reliability validation

## Ground Rules
- Always pick a task from `coordination/task_board.md` before editing files.
- Claim file locks before changing files.
- Leave a report artifact for every request in `coordination/reports/`.
- Keep commits scoped to one task when possible.
- Never rewrite another agent's in-progress files without explicit handoff.

## Lock Protocol
Use `scripts/swarm_coord.py`.

Acquire lock:
```bash
python scripts/swarm_coord.py lock acquire \
  --agent SP \
  --task TASK-001 \
  src/depth_anything_3/services/backend.py
```

Release lock:
```bash
python scripts/swarm_coord.py lock release \
  --agent SP \
  src/depth_anything_3/services/backend.py
```

List locks:
```bash
python scripts/swarm_coord.py lock list
```

Rules:
- Lock before editing.
- If a lock exists by another agent, do not edit that file.
- Split work into non-overlapping file sets whenever possible.

## Task Protocol
Add task:
```bash
python scripts/swarm_coord.py task add \
  --id TASK-001 \
  --title "Remote frame ingestion API" \
  --owner SP
```

Move task status:
```bash
python scripts/swarm_coord.py task update --id TASK-001 --status in_progress
python scripts/swarm_coord.py task update --id TASK-001 --status done
```

List tasks:
```bash
python scripts/swarm_coord.py task list
```

Status values:
- `todo`
- `in_progress`
- `blocked`
- `review`
- `done`

## Report Protocol
After each request, append a report:
```bash
python scripts/swarm_coord.py report add \
  --agent SP \
  --task TASK-001 \
  --summary "Added frame upload endpoint and queue wiring." \
  --files "src/depth_anything_3/services/backend.py,docs/VISUAL_SLAM_SYSTEM_DESIGN.md" \
  --next "TT validates API latency and error handling."
```

Artifacts are stored in `coordination/reports/` as timestamped markdown files.

## Merge and Conflict Avoidance
- Prefer one task per PR.
- Prefer one owner per file during a task.
- If conflicts happen:
  1. keep behavior changes from task owner
  2. preserve refactors only when tests stay green
  3. escalate to PM for final resolution

## Definition of Done
A task is done when:
- implementation is complete
- local validation for the touched scope is complete
- docs are updated if behavior changed
- report file is written
- locks are released
