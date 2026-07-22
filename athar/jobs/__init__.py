"""Durable job service (D13/D19).

- :mod:`athar.jobs.queue` — SQLite-backed queue (own DB file, WAL,
  ``UPDATE...RETURNING`` atomic claims), cancel flags, stale-job requeue.
- :mod:`athar.jobs.worker` — worker process loop + executors
  (``run_pipeline`` local executor; Kaggle executor seam per D13).
- :mod:`athar.jobs.service` — the facade the API layer talks to.

The typed event pipe is the run's ``events.jsonl``: the job row carries
``run_id`` and consumers tail the run's event log — no second event store.
"""
