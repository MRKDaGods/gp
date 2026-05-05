const GOOGLE_MAPS_DIR_BASE = "https://www.google.com/maps/dir/";
/** QR and deep links stay under this length when possible. */
const GOOGLE_MAPS_URL_MAX_LEN = 1800;

export type MapsPathCoordinate = { lat: number; lng: number };

function dedupeConsecutiveDirSegments(segments: string[]): string[] {
  const out: string[] = [];
  for (const s of segments) {
    if (out[out.length - 1] !== s) out.push(s);
  }
  return out;
}

function pickEvenlySpacedIndices(total: number, pickCount: number): number[] {
  if (pickCount <= 0 || total <= 0) return [];
  if (pickCount >= total) return Array.from({ length: total }, (_, i) => i);
  const idxs: number[] = [];
  for (let k = 0; k < pickCount; k++) {
    const idx = Math.round(((k + 1) / (pickCount + 1)) * (total + 1)) - 1;
    idxs.push(Math.min(Math.max(idx, 0), total - 1));
  }
  return [...new Set(idxs)].sort((a, b) => a - b);
}

/**
 * Multi-stop driving directions (numbered stops in Google Maps).
 * @see https://developers.google.com/maps/documentation/urls/get-started
 */
export function buildGoogleMapsDirectionsParamsUrl(coordSegments: string[]): string {
  const origin = coordSegments[0];
  const destination = coordSegments[coordSegments.length - 1];
  const params = new URLSearchParams();
  params.set("api", "1");
  params.set("origin", origin);
  params.set("destination", destination);
  params.set("travelmode", "driving");
  if (coordSegments.length > 2) {
    params.set("waypoints", coordSegments.slice(1, -1).join("|"));
  }
  return `${GOOGLE_MAPS_DIR_BASE}?${params.toString()}`;
}

function shortenGoogleMapsCoordSegments(segments: string[]): string[] {
  if (segments.length <= 1) return segments;
  if (buildGoogleMapsDirectionsParamsUrl(segments).length <= GOOGLE_MAPS_URL_MAX_LEN) {
    return segments;
  }
  if (segments.length === 2) return segments;

  const first = segments[0];
  const last = segments[segments.length - 1];
  const fullInner = segments.slice(1, -1);
  let innerCount = fullInner.length;

  while (innerCount >= 0) {
    const inner =
      innerCount === 0
        ? []
        : pickEvenlySpacedIndices(fullInner.length, innerCount).map((j) => fullInner[j]);
    const cand = dedupeConsecutiveDirSegments([first, ...inner, last]);
    if (buildGoogleMapsDirectionsParamsUrl(cand).length <= GOOGLE_MAPS_URL_MAX_LEN) return cand;
    innerCount -= 1;
  }
  return [first, last];
}

/**
 * QR / share: built from ordered path coordinates (e.g. full vehicle path).
 */
export function buildGoogleMapsPathShareUrl(pathSlice: MapsPathCoordinate[]): string | null {
  if (pathSlice.length === 0) return null;

  let segments = dedupeConsecutiveDirSegments(pathSlice.map((p) => `${p.lat},${p.lng}`));

  if (segments.length <= 1) {
    const p = pathSlice[0];
    const { lat, lng } = p;
    return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${lat},${lng}`)}`;
  }

  segments = shortenGoogleMapsCoordSegments(segments);

  if (segments.length <= 1) {
    const [pair] = segments;
    const parts = pair.split(",");
    const lat = Number(parts[0]);
    const lng = Number(parts[1]);
    if (Number.isFinite(lat) && Number.isFinite(lng)) {
      return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${lat},${lng}`)}`;
    }
  }

  return buildGoogleMapsDirectionsParamsUrl(segments);
}

/**
 * Driving directions to one point. Google Maps uses the user's current location (or prompts) when origin is omitted.
 */
export function buildGoogleMapsDirectionsToDestination(lat: number, lng: number): string {
  const params = new URLSearchParams();
  params.set("api", "1");
  params.set("destination", `${lat},${lng}`);
  params.set("travelmode", "driving");
  return `${GOOGLE_MAPS_DIR_BASE}?${params.toString()}`;
}
