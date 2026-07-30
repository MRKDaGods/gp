# Self-hosted MapLibre worker

`maplibre-gl-worker.mjs` + `maplibre-gl-shared.mjs` are copied verbatim
from `node_modules/maplibre-gl/dist/` (version-locked to the installed
maplibre-gl — currently **6.0.0**; recopy on upgrade).

Why: maplibre's `defaultWorkerUrl()` derives the worker URL from
`import.meta.url`, which is not an http(s) URL under Turbopack, so the
worker silently never spawns — vector sources then never load a single
tile while background + DOM markers still render (the basemap just looks
empty). `RunMap` points `maplibregl.setWorkerUrl()` here instead. Same
self-hosting policy as `public/maps/basemap.pmtiles` (air-gap: no CDN).
