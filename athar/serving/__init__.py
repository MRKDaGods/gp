"""Model serving: lifecycle registry, unified loader, cache, devices.

ONE loader path shared by the pipeline and the API (v1 duplicated model
construction between stage2 and serving). Checkpoints are content-addressed
(SHA-256); nothing loads a weights file except through a registry entry.
"""
