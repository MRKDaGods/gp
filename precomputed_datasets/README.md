# Precomputed dataset galleries

This folder holds **precomputed dataset galleries** — the full stages 0→4 output
of running the pipeline over a `dataset/<name>/` folder. They are reused as the
search "gallery" at Inference time (you upload a probe video, then search *within*
one of these precomputed datasets).

This directory is deliberately **separate from `outputs/`** (which holds ad-hoc /
probe runs) so galleries are easy to back up and copy between machines.

## Layout

One folder per precomputed dataset, named `dataset_precompute_<slug>` where
`<slug>` is the lowercased `dataset/` folder name:

```
precomputed_datasets/
  dataset_precompute_s01/
    stage0/ stage1/ stage2/ stage3/ stage4/
    run_context.json
  dataset_precompute_seif/
    ...
```

A gallery counts as "ready" when it has `stage1/tracklets_*.json`,
`stage2/embeddings.npy` + `stage2/embedding_index.json`, and
`stage4/global_trajectories.json`.

## How they get here

- **Automatically** — click *Process Dataset* in the app's **Dataset** tab, or
  pick a not-yet-processed dataset at **Inference** (it auto-processes inline).
  Both write here via the stable id `dataset_precompute_<slug>`.
- **Manually** — copy an existing `dataset_precompute_<name>` folder into here.
  The app discovers it on the next `/api/datasets` call; no restart needed.
