import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

const DEFAULT_PUBLIC_API_URL = "http://localhost:8004/api";

/** Turn fetch() failures into a short actionable message for stage banners / toasts. */
export function formatNetworkFailure(err: unknown): string {
  const base =
    typeof process !== "undefined" &&
    process.env?.NEXT_PUBLIC_API_URL &&
    String(process.env.NEXT_PUBLIC_API_URL).trim() !== ""
      ? String(process.env.NEXT_PUBLIC_API_URL)
      : DEFAULT_PUBLIC_API_URL;
  const raw = err instanceof Error ? err.message : String(err);
  if (/failed to fetch|network error|networkerror|load failed|network request failed/i.test(raw)) {
    return (
      `Cannot reach the API at ${base}. ` +
      "Start the backend (e.g. python start.py) or set NEXT_PUBLIC_API_URL to match the server."
    );
  }
  return raw;
}

export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let unitIndex = 0;
  let size = bytes;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function throttle<T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
}

// Camera colors for visualization
export const CAMERA_COLORS: Record<string, string> = {
  'c001': '#3b82f6', // blue
  'c002': '#8b5cf6', // purple
  'c003': '#ec4899', // pink
  'c004': '#f97316', // orange
  'c005': '#14b8a6', // teal
  'c006': '#eab308', // yellow
  'c007': '#22c55e', // green
  'c008': '#ef4444', // red
  'S01_c001': '#3b82f6',
  'S01_c002': '#8b5cf6',
  'S01_c003': '#ec4899',
  'S02_c006': '#f97316',
  'S02_c007': '#14b8a6',
  'S02_c008': '#eab308',
};

export function getCameraColor(cameraId: string): string {
  if (CAMERA_COLORS[cameraId]) {
    return CAMERA_COLORS[cameraId];
  }
  // Generate consistent color from camera ID
  let hash = 0;
  for (let i = 0; i < cameraId.length; i++) {
    hash = cameraId.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash % 360);
  return `hsl(${hue}, 70%, 50%)`;
}

// Class colors
export const CLASS_COLORS: Record<number, string> = {
  0: '#3b82f6',  // person - blue
  2: '#22c55e',  // car - green
  5: '#f97316',  // bus - orange
  7: '#8b5cf6',  // truck - purple
};

export function getClassColor(classId: number): string {
  return CLASS_COLORS[classId] || '#64748b';
}

export function getClassName(classId: number): string {
  const names: Record<number, string> = {
    0: 'Person',
    2: 'Car',
    5: 'Bus',
    7: 'Truck',
  };
  return names[classId] || 'Unknown';
}

// Bounding box utilities
export function bboxToStyle(
  bbox: { x1: number; y1: number; x2: number; y2: number },
  containerWidth: number,
  containerHeight: number,
  videoWidth: number,
  videoHeight: number
) {
  const scaleX = containerWidth / videoWidth;
  const scaleY = containerHeight / videoHeight;

  return {
    left: bbox.x1 * scaleX,
    top: bbox.y1 * scaleY,
    width: (bbox.x2 - bbox.x1) * scaleX,
    height: (bbox.y2 - bbox.y1) * scaleY,
  };
}

// Time range utilities
export function overlaps(
  range1: [number, number],
  range2: [number, number]
): boolean {
  return range1[0] <= range2[1] && range2[0] <= range1[1];
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// Array utilities
export function groupBy<T>(
  array: T[],
  keyFn: (item: T) => string
): Record<string, T[]> {
  return array.reduce((acc, item) => {
    const key = keyFn(item);
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(item);
    return acc;
  }, {} as Record<string, T[]>);
}

export function sortBy<T>(
  array: T[],
  keyFn: (item: T) => number | string,
  direction: 'asc' | 'desc' = 'asc'
): T[] {
  return [...array].sort((a, b) => {
    const aKey = keyFn(a);
    const bKey = keyFn(b);
    const comparison = aKey < bKey ? -1 : aKey > bKey ? 1 : 0;
    return direction === 'asc' ? comparison : -comparison;
  });
}

export function uniqueBy<T>(
  array: T[],
  keyFn: (item: T) => string | number
): T[] {
  const seen = new Set<string | number>();
  return array.filter(item => {
    const key = keyFn(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
