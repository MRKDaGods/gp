"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type SyntheticEvent,
} from "react";
import { cn } from "@/lib/utils";

type Buf = 0 | 1;

/**
 * Two stacked full-bleed images: the next URL loads in the hidden layer, then we swap.
 * Prevents blank flashes when changing `src` every frame (single &lt;img&gt; clears before decode).
 */
export function DoubleBufferedFrameImg({
  src,
  className,
  alt,
  imgDecoding = "sync",
}: {
  src: string;
  className?: string;
  alt: string;
  /** `async` avoids blocking the main thread during rapid frame flips (slightly softer sync). */
  imgDecoding?: "sync" | "async";
}) {
  const [[u0, u1], setUrls] = useState<[string, string]>(() => [src, src]);
  const [active, setActive] = useState<Buf>(0);
  const pendingRef = useRef<{ layer: Buf; gen: number } | null>(null);
  const genRef = useRef(0);

  useEffect(() => {
    const showing = active === 0 ? u0 : u1;
    if (src === showing) return;
    const next: Buf = active === 0 ? 1 : 0;
    genRef.current += 1;
    pendingRef.current = { layer: next, gen: genRef.current };
    setUrls(([a, b]) => (next === 0 ? [src, b] : [a, src]));
  }, [src, active, u0, u1]);

  const onLoad = useCallback((layer: Buf) => {
    const p = pendingRef.current;
    if (!p || p.layer !== layer || p.gen !== genRef.current) return;
    setActive(layer);
    pendingRef.current = null;
  }, []);

  return (
    <div className="absolute inset-0 z-0">
      <img
        src={u0}
        alt={active === 0 ? alt : ""}
        className={cn(
          className,
          "absolute inset-0 h-full w-full",
          active === 0 ? "z-[2] opacity-100" : "z-[1] opacity-0"
        )}
        draggable={false}
        decoding={imgDecoding}
        onLoad={() => onLoad(0)}
      />
      <img
        src={u1}
        alt={active === 1 ? alt : ""}
        className={cn(
          className,
          "absolute inset-0 h-full w-full",
          active === 1 ? "z-[2] opacity-100" : "z-[1] opacity-0"
        )}
        draggable={false}
        decoding={imgDecoding}
        onLoad={() => onLoad(1)}
      />
    </div>
  );
}

/**
 * Tracklet full-frame + bbox overlay. Parent should throttle frame picks (e.g. ~15–20/s) so
 * `src` does not change every React tick — that keeps decode + layout work bounded.
 */
export function TrackletFrameView({
  src,
  bbox,
  className,
  imgClassName,
}: {
  src: string;
  bbox: number[];
  className?: string;
  imgClassName?: string;
}) {
  const imgRef = useRef<HTMLImageElement>(null);

  const computeLayout = useCallback((el: HTMLImageElement | null) => {
    if (!el || el.naturalWidth <= 0 || el.naturalHeight <= 0) return null;
    const nw = el.naturalWidth;
    const nh = el.naturalHeight;
    const cw = el.clientWidth;
    const ch = el.clientHeight;
    if (cw <= 0 || ch <= 0) return null;
    const scale = Math.min(cw / nw, ch / nh);
    const dw = nw * scale;
    const dh = nh * scale;
    const ox = (cw - dw) / 2;
    const oy = (ch - dh) / 2;
    return { ox, oy, scale };
  }, []);

  const [layout, setLayout] = useState<{ ox: number; oy: number; scale: number } | null>(null);

  const syncLayout = useCallback(() => {
    const L = computeLayout(imgRef.current);
    setLayout((prev) => (L ? L : prev));
  }, [computeLayout]);

  useLayoutEffect(() => {
    syncLayout();
  }, [src, bbox, syncLayout]);

  useEffect(() => {
    const el = imgRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => syncLayout());
    ro.observe(el);
    return () => ro.disconnect();
  }, [src, syncLayout]);

  const onLoad = (e: SyntheticEvent<HTMLImageElement>) => {
    const L = computeLayout(e.currentTarget);
    setLayout((prev) => (L ? L : prev));
  };

  const bx = bbox;
  const bw = bx.length === 4 ? bx[2] - bx[0] : 0;
  const bh = bx.length === 4 ? bx[3] - bx[1] : 0;
  const valid = bx.length === 4 && bw > 1e-3 && bh > 1e-3;

  return (
    <div
      className={cn(
        "flex h-full min-h-0 w-full min-w-0 items-center justify-center bg-black",
        className
      )}
    >
      <div className="relative inline-block max-h-full max-w-full">
        <img
          ref={imgRef}
          src={src}
          alt=""
          className={cn(imgClassName, "block h-auto max-h-full w-auto max-w-full object-contain")}
          draggable={false}
          decoding="async"
          onLoad={onLoad}
        />
        {layout && valid && (
          <div
            className="pointer-events-none absolute z-10 rounded-sm border-2 border-green-400 shadow-[0_0_10px_rgba(34,197,94,0.85)]"
            style={{
              left: layout.ox + bx[0] * layout.scale,
              top: layout.oy + bx[1] * layout.scale,
              width: (bx[2] - bx[0]) * layout.scale,
              height: (bx[3] - bx[1]) * layout.scale,
            }}
          />
        )}
      </div>
    </div>
  );
}
