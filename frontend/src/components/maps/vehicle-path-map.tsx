"use client";

import { useEffect } from "react";
import type { LatLngTuple } from "leaflet";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";

type CameraPoint = { cameraId: string; lat: number; lng: number; label: string };

type PathPoint = { cameraId: string; lat: number; lng: number; tooltip: string; sequence: number };

interface VehiclePathMapProps {
  center: LatLngTuple;
  fitPoints: Array<{ lat: number; lng: number }>;
  cameraPoints: CameraPoint[];
  pathLatLngs: LatLngTuple[];
  pathPoints: PathPoint[];
  pathColor?: string;
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
  pathPoints,
  pathColor,
}: VehiclePathMapProps) {
  const strokeColor = pathColor ?? "#22c55e";

  return (
    <MapContainer className="h-full w-full" center={center} zoom={17} scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds points={fitPoints} />
      {cameraPoints.map((cam) => (
        <CircleMarker
          key={`cam-${cam.cameraId}`}
          center={[cam.lat, cam.lng]}
          radius={4}
          pathOptions={{ color: "#94a3b8", weight: 1, fillOpacity: 0.5 }}
        >
          <Tooltip direction="top" offset={[0, -6]}>
            {cam.label}
          </Tooltip>
        </CircleMarker>
      ))}
      {pathLatLngs.length > 1 && (
        <Polyline
          positions={pathLatLngs}
          pathOptions={{ color: strokeColor, weight: 4, opacity: 0.9 }}
        />
      )}
      {pathPoints.map((point) => (
        <CircleMarker
          key={`path-${point.cameraId}-${point.sequence}`}
          center={[point.lat, point.lng]}
          radius={7}
          pathOptions={{ color: strokeColor, weight: 2, fillOpacity: 0.9 }}
        >
          <Tooltip direction="top" offset={[0, -6]}>
            {point.tooltip}
          </Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
