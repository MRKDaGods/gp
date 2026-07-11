/**
 * Tolerant nearest-frame lookup for per-frame detection overlays.
 *
 * Detections are keyed by the frame ids tracking actually produced, which are NOT
 * always contiguous - a sampled/sparse run yields gaps like 0, 3, 6. The scrubber
 * and playback clock step by 1, so an exact-key lookup misses the in-between frames
 * and the bounding boxes flicker off. Snapping to the nearest produced frame (bounded
 * by the typical spacing) keeps boxes on screen without making them "stick" long after
 * the object is gone.
 */

/** Nearest key within `maxGap` of `frame`, via binary search. Null when nothing is close. */
export function nearestFrameWithin(sortedKeys: number[], frame: number, maxGap: number): number | null {
  const n = sortedKeys.length;
  if (n === 0) return null;
  if (frame <= sortedKeys[0]) return sortedKeys[0] - frame <= maxGap ? sortedKeys[0] : null;
  if (frame >= sortedKeys[n - 1]) return frame - sortedKeys[n - 1] <= maxGap ? sortedKeys[n - 1] : null;
  let lo = 0;
  let hi = n - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const k = sortedKeys[mid];
    if (k === frame) return k;
    if (k < frame) lo = mid + 1;
    else hi = mid - 1;
  }
  // lo = first key > frame, hi = last key < frame
  const above = sortedKeys[lo];
  const below = sortedKeys[hi];
  const dBelow = below != null ? frame - below : Infinity;
  const dAbove = above != null ? above - frame : Infinity;
  if (Math.min(dBelow, dAbove) > maxGap) return null;
  return dBelow <= dAbove ? below : above;
}

/** Typical spacing between frames - the snap tolerance for nearestFrameWithin. */
export function typicalFrameGap(sortedKeys: number[]): number {
  if (sortedKeys.length < 2) return 1;
  const gaps: number[] = [];
  for (let i = 1; i < sortedKeys.length; i += 1) gaps.push(sortedKeys[i] - sortedKeys[i - 1]);
  gaps.sort((a, b) => a - b);
  return Math.max(1, gaps[Math.floor(gaps.length / 2)] || 1);
}

/** Values for a frame, snapping to the nearest produced frame within tolerance. */
export function valuesForFrame<T>(
  cache: Map<number, T[]>,
  sortedKeys: number[],
  frame: number,
  maxGap: number
): T[] {
  const exact = cache.get(frame);
  if (exact) return exact;
  const key = nearestFrameWithin(sortedKeys, frame, maxGap);
  return key != null ? cache.get(key) ?? [] : [];
}
