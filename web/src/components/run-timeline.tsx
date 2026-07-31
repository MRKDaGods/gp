"use client";

import { useTranslations } from "next-intl";
import { Fragment, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CrossCameraGallery } from "@/components/cross-camera-gallery";
import type {
  TimelineIdentityOut,
  TimelineMemberOut,
  TimelineOut,
} from "@/lib/api";
import { API_URL } from "@/lib/client";

// Golden-angle hues: stable, well-separated colors keyed by global_id so an
// identity keeps its color across lanes, the map, and future views.
export function identityColor(globalId: number): string {
  return `hsl(${(globalId * 137.508) % 360} 65% 50%)`;
}

function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function tickStep(spanS: number): number {
  for (const step of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800]) {
    if (spanS / step <= 8) return step;
  }
  return 3600;
}

type Selection = { identity: TimelineIdentityOut; member: TimelineMemberOut };

// The player should never open empty — default to the strongest cross-camera
// story (or just the first identity) so there's always something playing.
function defaultSelection(timeline: TimelineOut): Selection | null {
  const withMembers = (identity: TimelineIdentityOut) =>
    identity.members.find((m) => m.clip_available) ?? identity.members[0];
  const crossing = timeline.identities
    .filter((i) => i.cross_camera)
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  const pick = crossing[0] ?? timeline.identities[0];
  if (!pick) return null;
  const member = withMembers(pick);
  return member ? { identity: pick, member } : null;
}

export function RunTimeline({
  runId,
  timeline,
}: {
  runId: string;
  timeline: TimelineOut;
}) {
  const t = useTranslations("timeline");
  const [selected, setSelected] = useState<Selection | null>(() =>
    defaultSelection(timeline),
  );

  // a real multi-camera run carries hundreds of single-camera identities;
  // the investigator's stars are the cross-camera ones, so dense runs
  // default to showing only those (toggleable)
  const [crossOnly, setCrossOnly] = useState(
    () =>
      timeline.identities.length > 30 &&
      timeline.identities.some((i) => i.cross_camera),
  );
  const visibleIdentities = crossOnly
    ? timeline.identities.filter((i) => i.cross_camera)
    : timeline.identities;

  // long tapes squeeze short sightings into slivers — default to a view
  // clamped around the activity when it covers under half the tape
  const spans = visibleIdentities.flatMap((identity) =>
    identity.members.filter((m) => m.start_s !== null && m.end_s !== null),
  );
  const activityStart = Math.min(...spans.map((m) => m.start_s as number), Infinity);
  const activityEnd = Math.max(...spans.map((m) => m.end_s as number), 0);
  const tapeEnd = Math.max(timeline.span_end_s, 1);
  const hasActivityWindow =
    spans.length > 0 && activityEnd - activityStart < tapeEnd / 2;
  const [fitActivity, setFitActivity] = useState(true);

  // a real photo of each camera's feed, not just its id — the first
  // thumbnailed sighting on that camera, from any identity
  const cameraPreview = useMemo(() => {
    const preview = new Map<string, TimelineMemberOut>();
    for (const identity of timeline.identities) {
      for (const member of identity.members) {
        if (member.has_thumbnail && !preview.has(member.camera_id)) {
          preview.set(member.camera_id, member);
        }
      }
    }
    return preview;
  }, [timeline.identities]);

  if (timeline.identities.length === 0) {
    return <p className="text-sm text-muted-foreground">{t("no_identities")}</p>;
  }

  const margin = Math.max((activityEnd - activityStart) * 0.1, 2);
  const viewStart =
    hasActivityWindow && fitActivity ? Math.max(activityStart - margin, 0) : 0;
  const viewEnd =
    hasActivityWindow && fitActivity ? activityEnd + margin : tapeEnd;
  const span = Math.max(viewEnd - viewStart, 1);
  const step = tickStep(span);
  const ticks: number[] = [];
  for (let v = Math.ceil(viewStart / step) * step; v <= viewEnd; v += step) {
    ticks.push(v);
  }
  const pct = (value: number) => ((value - viewStart) / span) * 100;

  // one lane per camera; members grouped by camera, carrying their identity
  const lanes = timeline.cameras.map((camera) => ({
    camera,
    entries: visibleIdentities.flatMap((identity) =>
      identity.members
        .filter(
          (member) =>
            member.camera_id === camera.camera_id &&
            member.start_s !== null &&
            member.end_s !== null,
        )
        .map((member) => ({ identity, member })),
    ),
  }));

  const crossCount = timeline.identities.filter((i) => i.cross_camera).length;
  const entityLabel = (entityClass: string) => {
    const key = `class_${entityClass}`;
    return t.has(key) ? t(key) : entityClass;
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-sm text-muted-foreground">
          {t("summary", {
            identities: timeline.identities.length,
            cross: crossCount,
          })}
        </p>
        {hasActivityWindow && (
          <button
            type="button"
            onClick={() => setFitActivity((v) => !v)}
            className="rounded-md border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
          >
            {fitActivity ? t("full_span") : t("fit_activity")}
          </button>
        )}
        {crossCount > 0 && (
          <button
            type="button"
            onClick={() => setCrossOnly((v) => !v)}
            className={`rounded-md border px-3 py-1.5 text-sm transition-colors hover:bg-muted ${
              crossOnly ? "border-primary text-primary" : ""
            }`}
          >
            {crossOnly ? t("show_all_identities") : t("cross_camera_only")}
          </button>
        )}
      </div>

      <CrossCameraGallery
        runId={runId}
        identities={timeline.identities}
        selected={selected}
        onSelect={setSelected}
      />

      {selected && <EvidencePlayer runId={runId} selection={selected} onSelect={setSelected} />}

      {/* the time axis reads left-to-right in both locales, like any chart */}
      <div dir="ltr" className="space-y-2">
        <div className="relative ms-40 h-4 text-[10px] text-muted-foreground">
          {ticks.map((tick) => (
            <span
              key={tick}
              className="absolute -translate-x-1/2"
              style={{ left: `${pct(tick)}%` }}
            >
              {formatClock(tick)}
            </span>
          ))}
        </div>
        {lanes.map(({ camera, entries }) => {
          const preview = cameraPreview.get(camera.camera_id);
          return (
            <div key={camera.camera_id} className="flex items-center gap-2.5">
              <div className="flex w-40 shrink-0 items-center justify-end gap-2 text-end text-sm">
                <div className="min-w-0">
                  <span className="font-mono">{camera.camera_id}</span>
                  {!camera.video_on_disk && (
                    <span
                      className="ms-1 text-muted-foreground"
                      title={t("video_missing")}
                    >
                      ⚠
                    </span>
                  )}
                </div>
                {preview ? (
                  // eslint-disable-next-line @next/next/no-img-element -- API-served evidence crop
                  <img
                    src={`${API_URL}/runs/${runId}/thumbs/${preview.camera_id}/${preview.track_id}`}
                    alt={camera.camera_id}
                    className="size-9 shrink-0 rounded border object-cover"
                    onError={(e) => {
                      e.currentTarget.style.visibility = "hidden";
                    }}
                  />
                ) : (
                  <div className="size-9 shrink-0 rounded border bg-muted" />
                )}
              </div>
              <div className="relative h-11 flex-1 overflow-hidden rounded-md bg-muted/60">
                {camera.scene_end_s !== null && (
                  <div
                    className="absolute inset-y-0 bg-muted"
                    style={{
                      left: `${pct(camera.scene_start_s)}%`,
                      width: `${((camera.scene_end_s - camera.scene_start_s) / span) * 100}%`,
                    }}
                  />
                )}
                {entries.map(({ identity, member }) => {
                  const start = member.start_s as number;
                  const end = member.end_s as number;
                  const mid = (start + end) / 2;
                  const dim =
                    selected !== null &&
                    selected.identity.global_id !== identity.global_id;
                  const isSelected =
                    selected?.member === member &&
                    selected.identity.global_id === identity.global_id;
                  return (
                    <Fragment key={`${identity.global_id}-${member.track_id}`}>
                      <button
                        type="button"
                        onClick={() => setSelected({ identity, member })}
                        title={`#${identity.global_id} ${entityLabel(identity.entity_class)} · ${formatClock(start)}–${formatClock(end)}`}
                        className={`absolute inset-y-1 min-w-[10px] rounded-sm transition-opacity ${
                          dim ? "opacity-25" : "opacity-90 hover:opacity-100"
                        } ${isSelected ? "ring-2 ring-foreground" : ""} ${
                          identity.cross_camera ? "border border-foreground/60" : ""
                        }`}
                        style={{
                          left: `${pct(start)}%`,
                          width: `${Math.max(((end - start) / span) * 100, 0.6)}%`,
                          backgroundColor: identityColor(identity.global_id),
                        }}
                      />
                      {member.has_thumbnail && (
                        // eslint-disable-next-line @next/next/no-img-element -- API-served evidence crop
                        <img
                          src={`${API_URL}/runs/${runId}/thumbs/${member.camera_id}/${member.track_id}`}
                          alt=""
                          onClick={() => setSelected({ identity, member })}
                          className={`absolute top-1/2 size-8 -translate-x-1/2 -translate-y-1/2 cursor-pointer rounded-full border-2 object-cover shadow transition-all hover:z-10 hover:scale-125 ${
                            dim ? "opacity-25" : "opacity-100"
                          }`}
                          style={{
                            left: `${pct(mid)}%`,
                            borderColor: identityColor(identity.global_id),
                          }}
                          onError={(e) => {
                            e.currentTarget.style.visibility = "hidden";
                          }}
                        />
                      )}
                    </Fragment>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const SPEED_OPTIONS = [0.5, 1, 2] as const;

// The centerpiece: always-visible player (never an empty "click something"
// state), a camera-angle switcher for the current identity, a light
// transport bar (play/pause, ±1s step, speed) on top of the native
// scrub bar, and an Export video button that downloads the whole
// cross-camera journey as one stitched MP4 (server-side concat).
function EvidencePlayer({
  runId,
  selection,
  onSelect,
}: {
  runId: string;
  selection: Selection;
  onSelect: (selection: Selection) => void;
}) {
  const t = useTranslations("timeline");
  const { identity, member } = selection;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [speed, setSpeed] = useState(1);
  const entityLabel = (entityClass: string) => {
    const key = `class_${entityClass}`;
    return t.has(key) ? t(key) : entityClass;
  };
  const clipUrl =
    member.clip_available && member.start_s !== null && member.end_s !== null
      ? `${API_URL}/runs/${runId}/clips/${member.camera_id}?start_s=${member.start_s}&end_s=${member.end_s}`
      : null;
  const exportUrl = `${API_URL}/runs/${runId}/identities/${identity.global_id}/export`;
  const hops = identity.members
    .filter((m) => m.start_s !== null)
    .sort((a, b) => (a.start_s as number) - (b.start_s as number));

  const step = (deltaS: number) => {
    const v = videoRef.current;
    if (v) v.currentTime = Math.max(0, v.currentTime + deltaS);
  };
  const setPlaybackSpeed = (value: number) => {
    setSpeed(value);
    if (videoRef.current) videoRef.current.playbackRate = value;
  };

  return (
    <div className="space-y-3 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="inline-block size-3.5 rounded-full"
            style={{ backgroundColor: identityColor(identity.global_id) }}
          />
          <span className="font-mono text-base font-semibold">
            #{identity.global_id}
          </span>
          <Badge variant="outline">{entityLabel(identity.entity_class)}</Badge>
          {identity.cross_camera && (
            <Badge variant="secondary">{t("cross_camera_badge")}</Badge>
          )}
          {identity.confidence !== null && (
            <span className="text-sm text-muted-foreground">
              {t("confidence")}: {(identity.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <a href={exportUrl} download>
          <Button type="button" size="sm">
            {t("export_video")}
          </Button>
        </a>
      </div>

      {hops.length > 1 && (
        <div dir="ltr" className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">{t("camera_angle")}:</span>
          {hops.map((m) => {
            const active = m === member;
            return (
              <button
                key={`${m.camera_id}-${m.track_id}`}
                type="button"
                onClick={() => onSelect({ identity, member: m })}
                className={`flex items-center gap-2 rounded-md border px-2 py-1.5 text-sm transition-colors ${
                  active
                    ? "border-primary bg-primary/10 font-medium"
                    : "hover:bg-muted"
                }`}
              >
                {m.has_thumbnail && (
                  // eslint-disable-next-line @next/next/no-img-element -- API-served evidence crop
                  <img
                    src={`${API_URL}/runs/${runId}/thumbs/${m.camera_id}/${m.track_id}`}
                    alt=""
                    className="size-8 rounded object-cover"
                  />
                )}
                <span className="font-mono">{m.camera_id}</span>
                <span className="text-xs text-muted-foreground">
                  {m.start_s !== null ? formatClock(m.start_s) : ""}
                </span>
              </button>
            );
          })}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_16rem]">
        <div className="space-y-2">
          {clipUrl ? (
            <>
              {/* key remounts the player when the selection changes */}
              <video
                key={clipUrl}
                ref={videoRef}
                src={clipUrl}
                controls
                autoPlay
                muted
                className="max-h-[28rem] w-full rounded-md border bg-black"
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button type="button" variant="outline" size="sm" onClick={() => step(-1)}>
                  {t("back_1s")}
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={() => step(1)}>
                  {t("forward_1s")}
                </Button>
                <div className="flex items-center gap-1">
                  {SPEED_OPTIONS.map((option) => (
                    <button
                      key={option}
                      type="button"
                      onClick={() => setPlaybackSpeed(option)}
                      className={`rounded-md border px-2 py-1 text-xs transition-colors ${
                        option === speed
                          ? "border-primary bg-primary/10 font-medium"
                          : "hover:bg-muted"
                      }`}
                    >
                      {option}x
                    </button>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">{t("clip_unavailable")}</p>
          )}
        </div>

        <div className="space-y-2">
          <div className="text-xs text-muted-foreground">
            {t("camera")}: <span className="font-mono">{member.camera_id}</span> ·{" "}
            {t("track")}: <span className="font-mono">{member.track_id}</span>
          </div>
          {Object.keys(identity.evidence).length > 0 && (
            <div className="space-y-1">
              <div className="text-xs font-medium">{t("evidence_terms")}</div>
              {Object.entries(identity.evidence).map(([term, value]) => (
                <div key={term} className="flex items-center gap-2 text-xs">
                  <span className="w-24 text-muted-foreground">
                    {t.has(`term_${term}`) ? t(`term_${term}`) : term}
                  </span>
                  <div className="h-1.5 w-24 overflow-hidden rounded bg-muted" dir="ltr">
                    <div
                      className="h-full bg-primary"
                      style={{ width: `${Math.min(value * 100, 100)}%` }}
                    />
                  </div>
                  <span className="font-mono">{value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
          {member.has_thumbnail && (
            // eslint-disable-next-line @next/next/no-img-element -- API-served evidence, not an optimizable asset
            <img
              src={`${API_URL}/runs/${runId}/thumbs/${member.camera_id}/${member.track_id}`}
              alt={`#${identity.global_id}`}
              className="max-h-40 w-fit rounded-md border"
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
