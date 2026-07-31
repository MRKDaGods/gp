"use client";

import { useLocale, useTranslations } from "next-intl";
import { useParams } from "next/navigation";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { EventStream } from "@/components/event-stream";
import { RunTimeline } from "@/components/run-timeline";
import { StatusBadge } from "@/components/status-badge";
import { Alert, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  getRunRunsRunIdGet,
  getTimelineRunsRunIdTimelineGet,
  type RunManifest,
  type TimelineOut,
} from "@/lib/api";
import { API_URL } from "@/lib/client";

// maplibre touches window at import time — client-only chunk
const RunMap = dynamic(
  () => import("@/components/run-map").then((m) => m.RunMap),
  { ssr: false },
);

// The four config layers, lowest -> highest precedence (D6). Anything the
// profile did not supply verbatim is visually flagged — that per-key
// provenance is the "which config actually ran" answer v1 never had.
const LAYER_STYLES: Record<string, string> = {
  profile_default: "",
  deployment: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
  case: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  run_override: "bg-destructive/10 text-destructive dark:bg-destructive/20",
};

export default function RunDetailPage() {
  const t = useTranslations("runs");
  const tl = useTranslations("timeline");
  const locale = useLocale();
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<RunManifest | null>(null);
  const [timeline, setTimeline] = useState<TimelineOut | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    getRunRunsRunIdGet({ path: { run_id: runId } }).then(
      ({ data, response }) => {
        if (!response?.ok || !data) setMissing(true);
        else setRun(data);
      },
    );
    getTimelineRunsRunIdTimelineGet({ path: { run_id: runId } }).then(
      ({ data }) => {
        if (data) setTimeline(data);  // 404 without package.report -> no cards
      },
    );
  }, [runId]);

  // a real photo of each camera's feed, not just its id — the first
  // thumbnailed sighting on that camera, from any identity
  const cameraPreview = useMemo(() => {
    const preview = new Map<string, { camera_id: string; track_id: number }>();
    for (const identity of timeline?.identities ?? []) {
      for (const member of identity.members) {
        if (member.has_thumbnail && !preview.has(member.camera_id)) {
          preview.set(member.camera_id, member);
        }
      }
    }
    return preview;
  }, [timeline]);

  if (missing) {
    return (
      <Alert variant="destructive">
        <AlertTitle>{t("not_found")}</AlertTitle>
      </Alert>
    );
  }
  if (run === null) {
    return <Skeleton className="h-64 w-full" />;
  }

  const artifacts = Object.values(run.artifacts ?? {});

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-lg font-bold" dir="ltr">
          {run.run_id}
        </h1>
        <StatusBadge status={run.status ?? "created"} />
        <Badge variant="outline">{run.role}</Badge>
        <span className="text-sm text-muted-foreground">
          {run.profile_name} ·{" "}
          {run.created_at ? new Date(run.created_at).toLocaleString(locale) : ""}
        </span>
        {"package.report" in (run.artifacts ?? {}) && (
          <a
            className="rounded-md border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
            href={`${API_URL}/runs/${run.run_id}/report.pdf?locale=${locale}`}
          >
            {t("report_pdf")}
          </a>
        )}
      </div>

      {run.error && (
        <Alert variant="destructive">
          <AlertTitle>
            {t("error_label")}: {run.error}
          </AlertTitle>
        </Alert>
      )}

      {timeline && (
        <Card>
          <CardHeader>
            <CardTitle>{tl("title")}</CardTitle>
          </CardHeader>
          <CardContent>
            <RunTimeline runId={runId} timeline={timeline} />
          </CardContent>
        </Card>
      )}

      {timeline && (
        <Card>
          <CardHeader>
            <CardTitle>{tl("map_title")}</CardTitle>
          </CardHeader>
          <CardContent>
            <RunMap timeline={timeline} />
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>{t("inputs")}</CardTitle>
          </CardHeader>
          <CardContent>
            {(run.inputs ?? []).length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("empty")}</p>
            ) : (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {(run.inputs ?? []).map((input) => {
                  const preview = cameraPreview.get(input.camera_id);
                  const onDisk = timeline?.cameras.find(
                    (c) => c.camera_id === input.camera_id,
                  )?.video_on_disk;
                  return (
                    <div
                      key={input.camera_id}
                      className="overflow-hidden rounded-md border"
                    >
                      {preview ? (
                        // eslint-disable-next-line @next/next/no-img-element -- API-served evidence crop
                        <img
                          src={`${API_URL}/runs/${run.run_id}/thumbs/${preview.camera_id}/${preview.track_id}`}
                          alt={input.camera_id}
                          className="aspect-video w-full object-cover"
                          onError={(e) => {
                            e.currentTarget.style.display = "none";
                          }}
                        />
                      ) : (
                        <div className="flex aspect-video w-full items-center justify-center bg-muted text-xs text-muted-foreground">
                          {t("no_preview")}
                        </div>
                      )}
                      <div className="space-y-0.5 p-2 text-xs">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono font-medium">
                            {input.camera_id}
                          </span>
                          {onDisk === false && (
                            <span
                              className="text-muted-foreground"
                              title={tl("video_missing")}
                            >
                              ⚠
                            </span>
                          )}
                        </div>
                        <div className="text-muted-foreground">
                          {input.duration_s ?? "—"}s · {input.fps ?? "—"} {t("fps")}
                        </div>
                        <div
                          className="truncate font-mono text-muted-foreground"
                          dir="ltr"
                          title={input.sha256}
                        >
                          {input.sha256.slice(0, 12)}…
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t("artifacts_title")}</CardTitle>
          </CardHeader>
          <CardContent>
            {artifacts.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("empty")}</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("artifacts")}</TableHead>
                    <TableHead>{t("producer")}</TableHead>
                    <TableHead>{t("rows")}</TableHead>
                    <TableHead />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {artifacts.map((artifact) => (
                    <TableRow key={artifact.name}>
                      <TableCell className="font-mono text-xs">
                        {artifact.name}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {artifact.producer}
                      </TableCell>
                      <TableCell>{artifact.row_count ?? "—"}</TableCell>
                      <TableCell>
                        <a
                          className="text-xs underline-offset-4 hover:underline"
                          href={`${API_URL}/runs/${run.run_id}/artifacts/${artifact.name}`}
                        >
                          {t("download")}
                        </a>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex flex-wrap items-center gap-3">
            {t("config")}
            {run.config && (
              <span
                className="font-mono text-xs font-normal text-muted-foreground"
                dir="ltr"
              >
                {t("config_hash")}: {run.config.config_hash.slice(0, 16)}…
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!run.config ? (
            <p className="text-sm text-muted-foreground">{t("no_config")}</p>
          ) : (
            <div className="max-h-96 overflow-y-auto rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("key")}</TableHead>
                    <TableHead>{t("value")}</TableHead>
                    <TableHead>{t("provenance")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(run.config.values)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([key, value]) => {
                      const layer = run.config?.provenance?.[key] ?? "";
                      return (
                        <TableRow key={key}>
                          <TableCell className="font-mono text-xs" dir="ltr">
                            {key}
                          </TableCell>
                          <TableCell className="font-mono text-xs" dir="ltr">
                            {JSON.stringify(value)}
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant="secondary"
                              className={LAYER_STYLES[layer]}
                            >
                              {layer}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t("events")}</CardTitle>
        </CardHeader>
        <CardContent>
          <EventStream url={`${API_URL}/runs/${run.run_id}/events/stream`} />
        </CardContent>
      </Card>
    </section>
  );
}
