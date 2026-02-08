# Locking Notes

Locking is file-path based and managed by `scripts/swarm_coord.py`.

- Acquire before editing.
- Release after validation.
- If lock conflict occurs, pick a different file set or wait for handoff.
