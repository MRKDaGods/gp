"use client";

import { useLocale, useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { EventStream } from "@/components/event-stream";
import { StatusBadge } from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  attachRunCasesCaseIdRunsPost,
  createCaseCasesPost,
  getJobJobsJobIdGet,
  listProfilesIngestProfilesGet,
  listRunsRunsGet,
  submitJobJobsPost,
  type IngestProfileOut,
  type Job,
  type RunSummary,
} from "@/lib/api";
import { API_URL } from "@/lib/client";
import { cn } from "@/lib/utils";

// v1's flow started at "Upload" and walked the user to a timeline. This
// wizard is that entry point for v2: goal -> footage -> pipeline -> run.

type Goal = "gallery" | "probe";

type FileRow = {
  file: File;
  cameraId: string;
  status: "pending" | "uploading" | "done" | "error";
  progress: number; // 0..100
  path?: string;
  sha256?: string;
  error?: string;
};

// XHR (not fetch) so big evidence files get a real progress bar.
function uploadWithProgress(
  file: File,
  cameraId: string,
  batchId: string | null,
  onProgress: (pct: number) => void,
): Promise<{ batch_id: string; path: string; sha256: string }> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    form.append("camera_id", cameraId);
    if (batchId) form.append("batch_id", batchId);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/ingest/upload`);
    xhr.withCredentials = true;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress((e.loaded / e.total) * 100);
    };
    xhr.onload = () => {
      if (xhr.status === 201) resolve(JSON.parse(xhr.responseText));
      else {
        try {
          reject(new Error(JSON.parse(xhr.responseText).detail ?? xhr.statusText));
        } catch {
          reject(new Error(`HTTP ${xhr.status}`));
        }
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.send(form);
  });
}

// "cam03_entrance.mp4" -> "cam03_entrance"; keep it a valid slug
function cameraIdFromFilename(name: string): string {
  const stem = name.replace(/\.[^.]+$/, "");
  return stem.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 64) || "cam";
}

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

export default function IngestPage() {
  const t = useTranslations("ingest");
  const locale = useLocale();
  const router = useRouter();

  const [goal, setGoal] = useState<Goal | null>(null);
  const [galleries, setGalleries] = useState<RunSummary[]>([]);
  const [galleryId, setGalleryId] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<IngestProfileOut[]>([]);
  const [profile, setProfile] = useState("multiclass");
  const [rows, setRows] = useState<FileRow[]>([]);
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [creatingCase, setCreatingCase] = useState(false);
  const batchRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listProfilesIngestProfilesGet().then(({ data }) => {
      if (data) setProfiles(data);
    });
    listRunsRunsGet({ query: { role: "gallery" } }).then(({ data }) =>
      setGalleries((data ?? []).filter((r) => r.status === "completed")),
    );
  }, []);

  // Poll the submitted job until it settles; the SSE stream below shows
  // the fine-grained pipeline events while this drives the state machine.
  useEffect(() => {
    if (!job || TERMINAL.has(job.status ?? "")) return;
    const timer = setInterval(() => {
      getJobJobsJobIdGet({ path: { job_id: job.job_id! } }).then(({ data }) => {
        if (data) setJob(data);
      });
    }, 2000);
    return () => clearInterval(timer);
  }, [job]);

  function addFiles(files: FileList | null) {
    if (!files) return;
    const next = Array.from(files).map<FileRow>((file) => ({
      file,
      cameraId: cameraIdFromFilename(file.name),
      status: "pending",
      progress: 0,
    }));
    setRows((prev) => [...prev, ...next]);
  }

  async function uploadAll() {
    setUploading(true);
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (row.status === "done") continue;
      setRows((prev) =>
        prev.map((r, j) => (j === i ? { ...r, status: "uploading", error: undefined } : r)),
      );
      try {
        const result = await uploadWithProgress(
          row.file,
          row.cameraId,
          batchRef.current,
          (pct) =>
            setRows((prev) =>
              prev.map((r, j) => (j === i ? { ...r, progress: pct } : r)),
            ),
        );
        batchRef.current = result.batch_id;
        setRows((prev) =>
          prev.map((r, j) =>
            j === i
              ? { ...r, status: "done", progress: 100, path: result.path, sha256: result.sha256 }
              : r,
          ),
        );
      } catch (err) {
        setRows((prev) =>
          prev.map((r, j) =>
            j === i ? { ...r, status: "error", error: String((err as Error).message) } : r,
          ),
        );
      }
    }
    setUploading(false);
  }

  async function startRun() {
    const videos = Object.fromEntries(
      rows.filter((r) => r.status === "done" && r.path).map((r) => [r.cameraId, r.path!]),
    );
    setSubmitting(true);
    const { data, response } = await submitJobJobsPost({
      body: {
        videos,
        profile,
        role: goal === "probe" ? "probe" : "gallery",
        executor: "local",
        overrides: [],
        priority: 0,
      },
    });
    setSubmitting(false);
    if (response?.ok && data) setJob(data);
  }

  // Probe done -> stitch a case together (gallery + probe attached) and
  // land the investigator in the search panel, exactly where v1's flow led.
  async function openAsCase() {
    if (!job?.run_id || !galleryId) return;
    setCreatingCase(true);
    const { data } = await createCaseCasesPost({
      body: { title: t("case_title", { date: new Date().toLocaleDateString(locale) }) },
    });
    if (data) {
      await attachRunCasesCaseIdRunsPost({
        path: { case_id: data.case_id },
        body: { run_id: galleryId },
      });
      await attachRunCasesCaseIdRunsPost({
        path: { case_id: data.case_id },
        body: { run_id: job.run_id },
      });
      router.push(`/${locale}/cases/${data.case_id}`);
    }
    setCreatingCase(false);
  }

  const uploaded = rows.filter((r) => r.status === "done");
  const cameraIdsValid =
    rows.length > 0 &&
    rows.every((r) => /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(r.cameraId)) &&
    new Set(rows.map((r) => r.cameraId)).size === rows.length;
  const readyToRun =
    uploaded.length === rows.length &&
    rows.length > 0 &&
    (goal === "gallery" || galleryId !== null);
  const jobDone = job?.status === "completed";
  const jobFailed = job?.status === "failed" || job?.status === "cancelled";

  // -- step 4: job running / finished --------------------------------------
  if (job) {
    return (
      <section className="mx-auto max-w-3xl space-y-6">
        <h1 className="text-xl font-bold">{t("processing_title")}</h1>
        <Card>
          <CardHeader>
            <CardTitle className="flex flex-wrap items-center gap-3">
              <span className="font-mono text-sm">{job.job_id}</span>
              <StatusBadge status={job.status ?? "queued"} />
              {job.run_id && (
                <span className="font-mono text-xs text-muted-foreground" dir="ltr">
                  {job.run_id}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {!jobDone && !jobFailed && (
              <p className="text-sm text-muted-foreground">{t("processing_hint")}</p>
            )}
            {jobFailed && (
              <p className="text-sm text-destructive">{t("processing_failed")}</p>
            )}
            {job.run_id && (
              <EventStream url={`${API_URL}/jobs/${job.job_id}/events/stream`} />
            )}
          </CardContent>
        </Card>
        {jobDone && job.run_id && (
          <div className="flex flex-wrap gap-3">
            <Button size="lg" onClick={() => router.push(`/${locale}/runs/${job.run_id}`)}>
              {t("open_timeline")}
            </Button>
            {goal === "probe" && galleryId && (
              <Button size="lg" variant="outline" onClick={openAsCase} disabled={creatingCase}>
                {t("open_as_case")}
              </Button>
            )}
          </div>
        )}
      </section>
    );
  }

  // -- steps 1..3 -----------------------------------------------------------
  return (
    <section className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-xl font-bold">{t("title")}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{t("subtitle")}</p>
      </div>

      {/* step 1: goal */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-muted-foreground">
          1 · {t("step_goal")}
        </h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {(
            [
              { key: "gallery", title: t("goal_gallery"), hint: t("goal_gallery_hint") },
              { key: "probe", title: t("goal_probe"), hint: t("goal_probe_hint") },
            ] as const
          ).map((option) => (
            <button
              key={option.key}
              type="button"
              onClick={() => setGoal(option.key)}
              className={cn(
                "rounded-lg border p-4 text-start transition-colors hover:bg-muted/50",
                goal === option.key && "border-primary bg-primary/5 ring-1 ring-primary",
              )}
            >
              <div className="font-medium">{option.title}</div>
              <div className="mt-1 text-sm text-muted-foreground">{option.hint}</div>
            </button>
          ))}
        </div>
        {goal === "probe" && (
          <div className="space-y-2 rounded-lg border p-4">
            <div className="text-sm font-medium">{t("pick_gallery")}</div>
            {galleries.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("no_galleries")}</p>
            ) : (
              <div className="space-y-2">
                {galleries.map((run) => (
                  <button
                    key={run.run_id}
                    type="button"
                    onClick={() => setGalleryId(run.run_id)}
                    className={cn(
                      "flex w-full flex-wrap items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm transition-colors hover:bg-muted/50",
                      galleryId === run.run_id &&
                        "border-primary bg-primary/5 ring-1 ring-primary",
                    )}
                  >
                    <span className="font-mono text-xs" dir="ltr">
                      {run.run_id}
                    </span>
                    <span className="flex items-center gap-2">
                      <Badge variant="outline">{run.profile_name}</Badge>
                      <span className="text-xs text-muted-foreground">
                        {new Date(run.created_at).toLocaleDateString(locale)}
                      </span>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* step 2: footage */}
      {goal && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-muted-foreground">
            2 · {t("step_footage")}
          </h2>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex w-full flex-col items-center gap-2 rounded-lg border-2 border-dashed p-8 text-center transition-colors hover:bg-muted/40"
          >
            <span className="text-3xl">🎞️</span>
            <span className="text-sm font-medium">{t("drop_hint")}</span>
            <span className="text-xs text-muted-foreground">{t("drop_formats")}</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".mp4,.avi,.mkv,.mov,.ts"
            className="hidden"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          {rows.length > 0 && (
            <div className="space-y-2">
              {rows.map((row, i) => (
                <div
                  key={`${row.file.name}-${i}`}
                  className="flex flex-wrap items-center gap-3 rounded-md border px-3 py-2"
                >
                  <span className="min-w-0 flex-1 truncate text-sm" dir="ltr">
                    {row.file.name}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {(row.file.size / 1e6).toFixed(1)} MB
                  </span>
                  <Input
                    value={row.cameraId}
                    dir="ltr"
                    onChange={(e) =>
                      setRows((prev) =>
                        prev.map((r, j) =>
                          j === i ? { ...r, cameraId: e.target.value } : r,
                        ),
                      )
                    }
                    disabled={row.status !== "pending"}
                    className="w-36 font-mono text-xs"
                    aria-label={t("camera_id")}
                  />
                  {row.status === "uploading" && (
                    <div className="h-1.5 w-24 overflow-hidden rounded bg-muted" dir="ltr">
                      <div
                        className="h-full bg-primary transition-all"
                        style={{ width: `${row.progress}%` }}
                      />
                    </div>
                  )}
                  {row.status === "done" && (
                    <span className="text-xs text-green-600 dark:text-green-400">
                      ✓ {row.sha256?.slice(0, 8)}…
                    </span>
                  )}
                  {row.status === "error" && (
                    <span className="text-xs text-destructive">{row.error}</span>
                  )}
                  {row.status === "pending" && (
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() =>
                        setRows((prev) => prev.filter((_, j) => j !== i))
                      }
                    >
                      ✕
                    </Button>
                  )}
                </div>
              ))}
              {!cameraIdsValid && (
                <p className="text-xs text-destructive">{t("camera_ids_invalid")}</p>
              )}
              {uploaded.length < rows.length && (
                <Button onClick={uploadAll} disabled={uploading || !cameraIdsValid}>
                  {uploading ? t("uploading") : t("upload_all", { count: rows.length })}
                </Button>
              )}
            </div>
          )}
        </div>
      )}

      {/* step 3: pipeline + go */}
      {goal && uploaded.length > 0 && uploaded.length === rows.length && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-muted-foreground">
            3 · {t("step_pipeline")}
          </h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {profiles.map((p) => (
              <button
                key={p.name}
                type="button"
                onClick={() => setProfile(p.name)}
                className={cn(
                  "rounded-lg border p-4 text-start transition-colors hover:bg-muted/50",
                  profile === p.name && "border-primary bg-primary/5 ring-1 ring-primary",
                )}
              >
                <div className="font-mono font-medium">{p.name}</div>
                <div className="mt-1 text-sm text-muted-foreground">{p.description}</div>
              </button>
            ))}
          </div>
          <Button size="lg" onClick={startRun} disabled={!readyToRun || submitting}>
            {submitting ? t("starting") : t("start_run")}
          </Button>
          {goal === "probe" && galleryId === null && (
            <p className="text-xs text-destructive">{t("gallery_required")}</p>
          )}
        </div>
      )}
    </section>
  );
}
