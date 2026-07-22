"use client";

import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";

// The API names its SSE frames after the pipeline event types, so a plain
// onmessage never fires — every known type gets an explicit listener. The
// server ends the stream after a terminal event; we close the EventSource
// then too, or the browser would reconnect against a finished stream.
const EVENT_TYPES = [
  "message",
  "stage_started",
  "stage_progress",
  "artifact_written",
  "stage_completed",
  "stage_skipped",
  "run_completed",
  "run_failed",
  "run_cancelled",
] as const;
const TERMINAL = new Set(["run_completed", "run_failed", "run_cancelled"]);
const MAX_EVENTS = 200;

type StreamEvent = { event?: string; stage?: string } & Record<string, unknown>;

function describe(data: StreamEvent): string {
  const parts: string[] = [];
  if (typeof data.stage === "string") parts.push(data.stage);
  for (const key of ["chunk", "progress", "name", "reason", "error"]) {
    const value = data[key];
    if (value !== undefined && value !== null) parts.push(`${key}=${value}`);
  }
  return parts.join("  ");
}

export function EventStream({ url }: Readonly<{ url: string }>) {
  const t = useTranslations("runs");
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [state, setState] = useState<"connecting" | "live" | "ended" | "error">(
    "connecting",
  );
  const endRef = useRef<HTMLDivElement>(null);

  // reset during render (not in the effect) when the tailed url changes
  const [prevUrl, setPrevUrl] = useState(url);
  if (prevUrl !== url) {
    setPrevUrl(url);
    setEvents([]);
    setState("connecting");
  }

  useEffect(() => {
    const source = new EventSource(url, { withCredentials: true });
    let ended = false;
    source.onopen = () => setState("live");
    const onEvent = (raw: MessageEvent) => {
      let data: StreamEvent;
      try {
        data = JSON.parse(raw.data);
      } catch {
        return;
      }
      setEvents((prev) => [...prev.slice(-(MAX_EVENTS - 1)), data]);
      if (data.event && TERMINAL.has(data.event)) {
        ended = true;
        source.close();
        setState("ended");
      }
    };
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, onEvent);
    }
    source.onerror = () => {
      if (!ended) {
        // completed runs drain then close without a terminal frame reaching
        // us twice; treat close-after-data as ended, not an error
        source.close();
        setState((s) => (s === "live" ? "ended" : "error"));
      }
    };
    return () => source.close();
  }, [url]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [events.length]);

  return (
    <div className="space-y-2">
      <Badge variant="outline">
        {state === "live" && t("live")}
        {state === "connecting" && t("connecting")}
        {state === "ended" && t("stream_ended")}
        {state === "error" && t("stream_error")}
      </Badge>
      <div className="max-h-64 overflow-y-auto rounded-md border bg-muted/30 p-3 font-mono text-xs">
        {events.length === 0 ? (
          <p className="text-muted-foreground">{t("no_events")}</p>
        ) : (
          events.map((data, index) => (
            <div key={index} className="flex gap-3 py-0.5" dir="ltr">
              <span className="shrink-0 font-medium">{data.event}</span>
              <span className="truncate text-muted-foreground">
                {describe(data)}
              </span>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
