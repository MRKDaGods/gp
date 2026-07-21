"""FastAPI application: thin routers over services over the run store.

Arrives in Phase 4 (after parity gates pass). Includes authN/authZ, case
ownership, and the tamper-evident audit log. No router touches the
filesystem directly — everything goes through the run store and job service.
"""
