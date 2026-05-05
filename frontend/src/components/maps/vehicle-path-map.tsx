"use client";

import { useEffect, useMemo } from "react";
import type { LatLngTuple } from "leaflet";
import L from "leaflet";
import { MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";

type CameraPoint = { cameraId: string; lat: number; lng: number; label: string };

/** Matches trajectory key logic in output-stage (`normalizeCameraId` + fallback). */
function registryPinKey(cameraId: string): string {
  const text = String(cameraId ?? "").trim();
  if (!text) return "";
  const matches = text.match(/\d+/g);
  if (!matches || matches.length === 0) return text;
  const num = Number(matches[matches.length - 1]);
  if (!Number.isFinite(num)) return text;
  return String(num);
}

function createRegistryPinIcon(fill: string, selected: boolean): L.DivIcon {
  const ring = selected
    ? "<circle cx=\"14\" cy=\"13\" r=\"11\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"3\"/>"
    : "";
  const svg = `
    <svg width="28" height="34" viewBox="0 0 28 34" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      ${ring}
      <path d="M14 2C8.48 2 4 6.37 4 11.73c0 6.92 9.18 17.07 9.53 17.44.26.27.61.41.97.41s.71-.14.97-.41C15.82 28.8 25 18.65 25 11.73 25 6.37 20.52 2 14 2z"
        fill="${fill}" stroke="rgba(255,255,255,0.95)" stroke-width="1.5" stroke-linejoin="round"/>
      <circle cx="14" cy="11.5" r="3.2" fill="rgba(255,255,255,0.92)"/>
    </svg>`;

  return L.divIcon({
    className: "vehicle-path-leaflet-pin",
    html: svg,
    iconSize: [28, 34],
    iconAnchor: [14, 33],
    tooltipAnchor: [0, -28],
  });
}

interface VehiclePathMapProps {
  center: LatLngTuple;
  fitPoints: Array<{ lat: number; lng: number }>;
  cameraPoints: CameraPoint[];
  /** Polyline vertices for the portion of the route to draw (often truncated to Maps destination). */
  pathLatLngs: LatLngTuple[];
  pathColor?: string;
  /** Keys from `registryPinKey` for cameras visited on the selected path (colored pins). */
  pathCameraKeys?: Set<string>;
  /** Highlight pin matching this registry key (current Maps destination camera). */
  highlightedDestinationCameraKey?: string | null;
  /** When set, pin click opens Google Maps driving directions (handled in parent). */
  onCameraMarkerClick?: (cameraId: string) => void;
}

function FitBounds({ points }: { points: Array<{ lat: number; lng: number }> }) {
  const map = useMap();

  useEffect(() => {
    if (!points.length) return;
    const bounds = points.map((p) => [p.lat, p.lng]) as LatLngTuple[];
    map.fitBounds(bounds, { padding: [24, 24], maxZoom: 18 });
  }, [map, points]);

  return null;
}

export default function VehiclePathMap({
  center,
  fitPoints,
  cameraPoints,
  pathLatLngs,
  pathColor,
  pathCameraKeys,
  highlightedDestinationCameraKey,
  onCameraMarkerClick,
}: VehiclePathMapProps) {
  const strokeColor = pathColor ?? "#22c55e";
  const mutedPinFill = "#64748b";

  const pinIcons = useMemo(() => {
    const cache = new Map<string, L.DivIcon>();
    const keyFor = (fill: string, sel: boolean) => `${fill}-${sel ? "1" : "0"}`;
    return (fill: string, selected: boolean) => {
      const k = keyFor(fill, selected);
      let icon = cache.get(k);
      if (!icon) {
        icon = createRegistryPinIcon(fill, selected);
        cache.set(k, icon);
      }
      return icon;
    };
  }, []);

  return (
    <MapContainer className="h-full w-full" center={center} zoom={17} scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds points={fitPoints} />
      {pathLatLngs.length > 1 && (
        <Polyline
          positions={pathLatLngs}
          pathOptions={{ color: strokeColor, weight: 4, opacity: 0.9 }}
        />
      )}
      {cameraPoints.map((cam) => {
        const key = registryPinKey(cam.cameraId);
        const onPath = pathCameraKeys?.has(key) ?? false;
        const isDest =
          Boolean(highlightedDestinationCameraKey) &&
          highlightedDestinationCameraKey === key;
        const fill = onPath ? strokeColor : mutedPinFill;
        const clickEnabled = Boolean(onCameraMarkerClick);

        return (
          <Marker
            key={`cam-${cam.cameraId}`}
            position={[cam.lat, cam.lng]}
            icon={pinIcons(fill, isDest)}
            eventHandlers={
              clickEnabled
                ? {
                    click: () => {
                      onCameraMarkerClick?.(cam.cameraId);
                    },
                  }
                : undefined
            }
          >
            <Tooltip direction="top" offset={[0, -8]}>
              {clickEnabled ? (
                <>
                  {cam.label}
                  <span className="block text-[10px] opacity-90">
                    {onPath
                      ? "Click for Google Maps directions along the track to here"
                      : "Click for directions to this camera (from your location in Maps)"}
                  </span>
                </>
              ) : (
                cam.label
              )}
            </Tooltip>
          </Marker>
        );
      })}
    </MapContainer>
  );
}
