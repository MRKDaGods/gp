"use client";

import * as maplibregl from "maplibre-gl";
import { useTheme } from "next-themes";
import { useTranslations } from "next-intl";
import { Protocol } from "pmtiles";
import { useEffect, useRef, useState } from "react";
import { identityColor } from "@/components/run-timeline";
import {
  cameraLocationsCamerasLocationsGet,
  type CameraLocationOut,
  type TimelineOut,
} from "@/lib/api";
import "maplibre-gl/dist/maplibre-gl.css";

// AIR-GAP: everything the map loads is same-origin — the PMTiles basemap
// from /maps/basemap.pmtiles (see scripts/fetch_basemap.py) and camera
// coordinates from the API. No tile server, no glyph/sprite CDN (the style
// has no text layers, so no glyphs are ever requested).
const BASEMAP_URL = "/maps/basemap.pmtiles";

let protocolRegistered = false;
function ensurePmtilesProtocol() {
  if (!protocolRegistered) {
    // Self-hosted worker: maplibre's defaultWorkerUrl() derives the worker
    // URL from import.meta.url, which is not an http(s) URL under
    // Turbopack — the worker then silently never spawns and vector
    // sources never load a single tile (background + DOM markers still
    // render, so the failure looks like a subtly empty basemap). The
    // worker + its shared chunk are copied into public/maplibre/ (same
    // self-hosting policy as the pmtiles basemap; version-locked to the
    // installed maplibre-gl — recopy from node_modules on upgrade).
    maplibregl.setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");
    maplibregl.addProtocol("pmtiles", new Protocol().tile);
    protocolRegistered = true;
  }
}

// Minimal protomaps-schema style; two palettes keyed by app theme.
function buildStyle(dark: boolean, withBasemap: boolean): maplibregl.StyleSpecification {
  const palette = dark
    ? {
        background: "#1c1f26", earth: "#23272f", water: "#182430",
        landuse: "#252b30", roads: "#3a4149", buildings: "#2c323b",
      }
    : {
        background: "#e8e6e1", earth: "#f1efe9", water: "#a5c8e1",
        landuse: "#e5e9df", roads: "#ffffff", buildings: "#d9d5cc",
      };
  const style: maplibregl.StyleSpecification = {
    version: 8,
    sources: {},
    layers: [
      {
        id: "background",
        type: "background",
        paint: { "background-color": palette.background },
      },
    ],
  };
  if (!withBasemap) return style;
  style.sources.basemap = {
    type: "vector",
    // absolute URL: maplibre normalizes source URLs before the custom
    // protocol sees them, and a triple-slash relative form gets mangled
    url: `pmtiles://${window.location.origin}${BASEMAP_URL}`,
    attribution: "© OpenStreetMap contributors, Protomaps",
  };
  const vectorLayer = (
    id: string,
    sourceLayer: string,
    type: "fill" | "line",
    paint: Record<string, unknown>,
  ) =>
    ({
      id,
      type,
      source: "basemap",
      "source-layer": sourceLayer,
      paint,
    }) as unknown as maplibregl.LayerSpecification;
  style.layers.push(
    vectorLayer("earth", "earth", "fill", { "fill-color": palette.earth }),
    vectorLayer("landuse", "landuse", "fill", { "fill-color": palette.landuse }),
    vectorLayer("water", "water", "fill", { "fill-color": palette.water }),
    vectorLayer("roads", "roads", "line", {
      "line-color": palette.roads,
      "line-width": 1.5,
    }),
    vectorLayer("buildings", "buildings", "fill", {
      "fill-color": palette.buildings,
    }),
  );
  return style;
}

function pathsGeojson(
  timeline: TimelineOut,
  locations: Record<string, CameraLocationOut>,
): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];
  for (const identity of timeline.identities) {
    if (!identity.cross_camera) continue;
    const ordered = [...identity.members]
      .filter((m) => m.start_s !== null && locations[m.camera_id])
      .sort((a, b) => (a.start_s as number) - (b.start_s as number));
    const coords: [number, number][] = [];
    for (const member of ordered) {
      const loc = locations[member.camera_id];
      const point: [number, number] = [loc.lng, loc.lat];
      const last = coords[coords.length - 1];
      if (!last || last[0] !== point[0] || last[1] !== point[1]) {
        coords.push(point);
      }
    }
    if (coords.length >= 2) {
      features.push({
        type: "Feature",
        properties: {
          color: identityColor(identity.global_id),
          global_id: identity.global_id,
        },
        geometry: { type: "LineString", coordinates: coords },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

export function RunMap({ timeline }: { timeline: TimelineOut }) {
  const t = useTranslations("timeline");
  const { resolvedTheme } = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [locations, setLocations] =
    useState<Record<string, CameraLocationOut> | null>(null);
  const [hasBasemap, setHasBasemap] = useState<boolean | null>(null);

  useEffect(() => {
    cameraLocationsCamerasLocationsGet().then(({ data }) => {
      setLocations(data?.cameras ?? {});
    });
    fetch(BASEMAP_URL, { method: "HEAD" })
      .then((r) =>
        setHasBasemap(
          r.ok && !(r.headers.get("content-type") ?? "").includes("text/html"),
        ),
      )
      .catch(() => setHasBasemap(false));
  }, []);

  const runCameras = timeline.cameras.map((c) => c.camera_id);
  const located = locations
    ? runCameras.filter((cam) => locations[cam])
    : [];
  const dark = resolvedTheme === "dark";

  useEffect(() => {
    if (
      locations === null ||
      hasBasemap === null ||
      located.length === 0 ||
      !containerRef.current ||
      mapRef.current
    ) {
      return;
    }
    ensurePmtilesProtocol();
    const bounds = new maplibregl.LngLatBounds();
    for (const cam of located) {
      bounds.extend([locations[cam].lng, locations[cam].lat]);
    }
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: buildStyle(dark, hasBasemap),
      bounds,
      fitBoundsOptions: { padding: 60, maxZoom: 17 },
      attributionControl: hasBasemap ? { compact: true } : false,
    });
    mapRef.current = map;
    if (process.env.NODE_ENV === "development") {
      (window as unknown as Record<string, unknown>).__atharMap = map;
    }
    map.on("error", (e) => {
      console.error("map error:", e.error?.message ?? e);
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }));

    for (const cam of located) {
      const el = document.createElement("div");
      el.className =
        "flex items-center gap-1 rounded-full border border-foreground/30 bg-background/90 px-2 py-0.5 text-[10px] font-mono shadow";
      el.textContent = cam;
      new maplibregl.Marker({ element: el })
        .setLngLat([locations[cam].lng, locations[cam].lat])
        .addTo(map);
    }

    // style.load fires after the initial style AND every theme setStyle —
    // runtime sources/layers are dropped on restyle, so re-add them here
    map.on("style.load", () => {
      if (map.getSource("identity-paths")) return;
      map.addSource("identity-paths", {
        type: "geojson",
        data: pathsGeojson(timeline, locations),
      });
      map.addLayer({
        id: "identity-paths",
        type: "line",
        source: "identity-paths",
        paint: {
          "line-color": ["get", "color"],
          "line-width": 3,
          "line-opacity": 0.85,
          "line-dasharray": [2, 1],
        },
      });
    });
    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- rebuild only when data readiness flips
  }, [locations, hasBasemap, located.length]);

  // theme flips restyle the existing map in place
  useEffect(() => {
    if (mapRef.current && hasBasemap !== null) {
      mapRef.current.setStyle(buildStyle(dark, hasBasemap), { diff: false });
    }
  }, [dark, hasBasemap]);

  if (locations !== null && located.length === 0) {
    return <p className="text-sm text-muted-foreground">{t("no_coordinates")}</p>;
  }
  return (
    <div className="space-y-2">
      {hasBasemap === false && (
        <p className="text-xs text-muted-foreground">{t("no_basemap")}</p>
      )}
      <div ref={containerRef} dir="ltr" className="h-96 w-full rounded-md border" />
    </div>
  );
}
