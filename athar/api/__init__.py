"""FastAPI application: thin routers over services over the stores (D19).

- ``app.py`` — factory (``create_app``); ``athar serve`` runs it.
- ``routers/`` — auth, runs, jobs, models, search, audit.
- Server-side sessions + argon2 (``security.py``), ordered RBAC
  (``deps.py``), hash-chained audit log (``audit.py``).

No router touches the filesystem directly — everything goes through the
run store, job service, and lifecycle registry.
"""
