"""Run contracts: manifest, resolved-config provenance, artifact store.

The manifest is the API between the pipeline, the backend, and the frontend.
No subsystem may guess paths, scan directories for meaning, or parse run
semantics out of folder names — if it isn't in the manifest, it doesn't exist.
"""
