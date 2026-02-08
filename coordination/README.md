# Coordination Workspace

Shared coordination assets for multi-agent development.

## Files
- `task_board.md`: task backlog and current assignment.
- `reports/`: per-request execution logs.
- `locks/`: lock protocol notes.
- `templates/`: optional templates for manual notes.

## Operational Notes
- Runtime lock state is stored in `coordination/swarm_state.json` (local state).
- Use `scripts/swarm_coord.py` for all lock/task/report actions.
