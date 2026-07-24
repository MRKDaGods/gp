# Offline basemap

The map view loads `basemap.pmtiles` from this directory (same-origin —
the deployed app never fetches tiles from the network). The file is
site-specific and not committed; provision it on a connected admin box
with:

```bash
python scripts/fetch_basemap.py
```

which slices the Protomaps daily OpenStreetMap build around the extent of
`configs/camera_locations.json`, then carry the output here alongside the
app build. Without the file the map still renders camera markers and
identity paths over a blank canvas.
