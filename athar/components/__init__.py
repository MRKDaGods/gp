"""Pluggable pipeline components: protocols + registry.

Every algorithmic slot in the pipeline (detector, tracker, embedder, score
term, solver, spatial model, …) is a registered component created from typed
config. Profiles select components by name; investigators' "advanced options"
UI is generated from each component's config schema.
"""
