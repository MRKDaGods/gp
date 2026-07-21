"""Durable job service: SQLite-backed queue, worker subprocesses, typed
event ingestion, cancel/resume. Executor-agnostic (local | kaggle) per D13.
Arrives in Phase 4; the pipeline runner only depends on the event contract.
"""
