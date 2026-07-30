# Shorouk dataset rebuild

Rebuilds the local `data/raw/shorouk` layout on Kaggle from the legacy upload
[gumfreddy/seif-dataset](https://www.kaggle.com/datasets/gumfreddy/seif-dataset)
and publishes it as a new public dataset **mrkdagods/shorouk-dataset**.

Why it exists: the legacy upload holds raw Hikvision DVR exports named
`D{channel}_{timestamp}.mp4` that are really MPEG-PS streams, and the
`D` channel number is not the physical camera id. The kernel sniffs the real
container, reads the `Camera XX` OSD digits by **glyph template matching**
(templates harvested from the ground-truth local videos and embedded in the
script; tesseract misreads this blocky font - 7 as 2, 1 as 0 - while the
templates scored 56/56 locally), keeps the latest ~1 GB rollover segment per
camera (the four "duplicate" D-ids are continuations, not dupes), trims
every camera to the synchronized window `[T0, T0+1295.08s]` where
`T0 = max(segment start) = 17:40:51` DVR time, and re-encodes to HEVC
1080p25 ~1.7 Mbps (NVENC), emitting `c0XX/vdo.mp4` +
`camera_coordinates.json`.

Self-checks per output: the frame-0 OSD digits must template-match the same
camera id (hard error), and the frame-0 OSD clock is OCR'd warn-only (should
land in the 17:40:49-56 band of per-camera clock skew seen locally).
Templates can be regenerated with the local build script if the geometry
ever changes (harvests digit cells from `data/raw/shorouk` frame 9/250).

## Run

```
kaggle kernels push -p scripts/kaggle/shorouk_rebuild
```

One-time setup in the Kaggle notebook editor (Add-ons > Secrets): attach
`KAGGLE_USERNAME` and `KAGGLE_KEY` for the `mrkdagods` account and toggle them
on for this kernel — the publish step needs them. GPU + internet must be on
(the metadata requests both).

## Outputs

- Dataset `mrkdagods/shorouk-dataset`: `c017..c032/vdo.mp4` (14 cams, no
  24/25) + `camera_coordinates.json`
- `rebuild_report.json` in the kernel output: per-file audit (magic bytes,
  ffprobe format, OCR result, dedup decision, output probe)

The kernel refuses to publish if any OCR fails or a roster camera has no
mapped source — check the report, then rerun.
