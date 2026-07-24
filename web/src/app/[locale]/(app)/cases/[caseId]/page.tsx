"use client";

import { useLocale, useTranslations } from "next-intl";
import { useParams } from "next/navigation";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Alert, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
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
  attachRunCasesCaseIdRunsPost,
  decideHypothesisCasesCaseIdTargetsTargetIdHypothesesHypothesisIdDecidePost,
  detachRunCasesCaseIdRunsRunIdDelete,
  createTargetCasesCaseIdTargetsPost,
  getCaseCasesCaseIdGet,
  listRunsRunsGet,
  proposeHypothesisCasesCaseIdTargetsTargetIdHypothesesPost,
  searchSearchPost,
  updateCaseCasesCaseIdPatch,
  type CaseDetail,
  type HypothesisOut,
  type RunSummary,
  type SearchResponse,
  type TargetOut,
} from "@/lib/api";
import { API_URL } from "@/lib/client";

export default function CaseWorkspacePage() {
  const t = useTranslations("cases");
  const locale = useLocale();
  const { caseId } = useParams<{ caseId: string }>();
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [missing, setMissing] = useState(false);

  const refresh = useCallback(() => {
    getCaseCasesCaseIdGet({ path: { case_id: caseId } }).then(
      ({ data, response }) => {
        if (!response?.ok || !data) {
          setMissing(true);
        } else {
          setDetail(data);
        }
      },
    );
  }, [caseId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (missing) {
    return (
      <Alert variant="destructive">
        <AlertTitle>{t("not_found")}</AlertTitle>
      </Alert>
    );
  }
  if (detail === null) {
    return <Skeleton className="h-64 w-full" />;
  }

  async function toggleStatus() {
    if (!detail) return;
    await updateCaseCasesCaseIdPatch({
      path: { case_id: detail.case_id },
      body: { status: detail.status === "open" ? "closed" : "open" },
    });
    refresh();
  }

  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold">{detail.title}</h1>
          <StatusBadge status={detail.status} />
          <span className="text-sm text-muted-foreground">
            {t("owner")}: {detail.owner}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <a
            className="rounded-md border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
            href={`${API_URL}/cases/${detail.case_id}/report.pdf?locale=${locale}`}
          >
            {t("report_pdf")}
          </a>
          <Button variant="outline" size="sm" onClick={toggleStatus}>
            {detail.status === "open" ? t("close_case") : t("reopen_case")}
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <EvidenceRuns detail={detail} onChanged={refresh} />
        <Targets detail={detail} onChanged={refresh} />
      </div>

      <SearchPanel detail={detail} locale={locale} onChanged={refresh} />
    </section>
  );
}

function EvidenceRuns({
  detail,
  onChanged,
}: Readonly<{ detail: CaseDetail; onChanged: () => void }>) {
  const t = useTranslations("cases");
  const [available, setAvailable] = useState<RunSummary[]>([]);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    listRunsRunsGet().then(({ data }) => setAvailable(data ?? []));
  }, [detail.runs.length]);

  const attached = useMemo(
    () => new Set(detail.runs.map((r) => r.run_id)),
    [detail.runs],
  );
  const attachable = available.filter((r) => !attached.has(r.run_id));

  async function attach() {
    if (!picked) return;
    await attachRunCasesCaseIdRunsPost({
      path: { case_id: detail.case_id },
      body: { run_id: picked },
    });
    setPicked(null);
    onChanged();
  }

  async function detach(runId: string) {
    await detachRunCasesCaseIdRunsRunIdDelete({
      path: { case_id: detail.case_id, run_id: runId },
    });
    onChanged();
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("evidence")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {detail.runs.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("no_runs")}</p>
        ) : (
          <ul className="space-y-2">
            {detail.runs.map((run) => (
              <li
                key={run.run_id}
                className="flex items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm"
              >
                <span className="truncate font-mono text-xs">{run.run_id}</span>
                <span className="flex items-center gap-2">
                  <Badge variant="outline">{run.role}</Badge>
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={() => detach(run.run_id)}
                  >
                    {t("detach")}
                  </Button>
                </span>
              </li>
            ))}
          </ul>
        )}
        <div className="flex items-end gap-2">
          <div className="min-w-0 flex-1 space-y-2">
            <Label>{t("attach_run")}</Label>
            <Select value={picked} onValueChange={(v) => setPicked(v as string)}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder={t("pick_run")} />
              </SelectTrigger>
              <SelectContent>
                {attachable.map((run) => (
                  <SelectItem key={run.run_id} value={run.run_id}>
                    {run.run_id} ({run.role})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button size="sm" onClick={attach} disabled={!picked}>
            {t("attach")}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function Targets({
  detail,
  onChanged,
}: Readonly<{ detail: CaseDetail; onChanged: () => void }>) {
  const t = useTranslations("cases");
  const [label, setLabel] = useState("");

  async function addTarget(event: FormEvent) {
    event.preventDefault();
    await createTargetCasesCaseIdTargetsPost({
      path: { case_id: detail.case_id },
      body: { label },
    });
    setLabel("");
    onChanged();
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("targets")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={addTarget} className="flex items-end gap-2">
          <div className="flex-1 space-y-2">
            <Label htmlFor="target-label">{t("new_target")}</Label>
            <Input
              id="target-label"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              required
              maxLength={256}
            />
          </div>
          <Button type="submit" size="sm">
            {t("add_target")}
          </Button>
        </form>
        {detail.targets.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("no_targets")}</p>
        ) : (
          detail.targets.map((target) => (
            <TargetCard
              key={target.target_id}
              caseId={detail.case_id}
              target={target}
              onChanged={onChanged}
            />
          ))
        )}
      </CardContent>
    </Card>
  );
}

function TargetCard({
  caseId,
  target,
  onChanged,
}: Readonly<{ caseId: string; target: TargetOut; onChanged: () => void }>) {
  const t = useTranslations("cases");
  const tStatus = useTranslations("statuses");

  async function decide(hyp: HypothesisOut, status: "confirmed" | "rejected") {
    await decideHypothesisCasesCaseIdTargetsTargetIdHypothesesHypothesisIdDecidePost(
      {
        path: {
          case_id: caseId,
          target_id: target.target_id,
          hypothesis_id: hyp.hypothesis_id,
        },
        body: { status },
      },
    );
    onChanged();
  }

  return (
    <div className="space-y-3 rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <span className="font-semibold">{target.label}</span>
        <span className="text-xs text-muted-foreground">
          {target.created_by}
        </span>
      </div>
      <div className="space-y-1">
        <p className="text-xs font-medium text-muted-foreground">
          {t("members")}
        </p>
        {target.members.length === 0 ? (
          <p className="text-xs text-muted-foreground">{t("no_members")}</p>
        ) : (
          <div className="flex flex-wrap gap-1">
            {target.members.map((m) => (
              <Badge
                key={`${m.run_id}/${m.camera_id}/${m.track_id}`}
                variant="secondary"
                className="font-mono text-xs"
              >
                {m.camera_id}#{m.track_id}
              </Badge>
            ))}
          </div>
        )}
      </div>
      <Separator />
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">
          {t("hypotheses")}
        </p>
        {target.hypotheses.length === 0 ? (
          <p className="text-xs text-muted-foreground">{t("no_hypotheses")}</p>
        ) : (
          target.hypotheses.map((hyp) => (
            <div
              key={hyp.hypothesis_id}
              className="flex flex-wrap items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm"
            >
              <span className="flex items-center gap-2">
                <Badge variant="outline">{hyp.kind}</Badge>
                <span className="font-mono text-xs">
                  {hyp.camera_id}#{hyp.track_id}
                </span>
                <span className="text-xs text-muted-foreground">
                  {hyp.probability != null
                    ? `${t("probability")} ${(hyp.probability * 100).toFixed(1)}%`
                    : `${t("score")} ${hyp.raw_score.toFixed(3)} (${t("uncalibrated")})`}
                </span>
              </span>
              <span className="flex items-center gap-2">
                {hyp.status === "proposed" ? (
                  <>
                    <span className="text-xs text-muted-foreground">
                      {t("proposed_by", { name: hyp.proposed_by })}
                    </span>
                    <Button size="xs" onClick={() => decide(hyp, "confirmed")}>
                      {t("confirm")}
                    </Button>
                    <Button
                      size="xs"
                      variant="destructive"
                      onClick={() => decide(hyp, "rejected")}
                    >
                      {t("reject")}
                    </Button>
                  </>
                ) : (
                  <>
                    <StatusBadge status={hyp.status} />
                    <span className="text-xs text-muted-foreground">
                      {t("decided_by", {
                        status: tStatus(hyp.status),
                        name: hyp.decided_by ?? "",
                      })}
                    </span>
                  </>
                )}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function SearchPanel({
  detail,
  locale,
  onChanged,
}: Readonly<{ detail: CaseDetail; locale: string; onChanged: () => void }>) {
  const t = useTranslations("cases");
  const galleries = detail.runs.filter((r) => r.role === "gallery");
  const probes = detail.runs.filter((r) => r.role === "probe");
  const [galleryId, setGalleryId] = useState<string | null>(null);
  const [probeId, setProbeId] = useState<string | null>(null);
  const [topK, setTopK] = useState(10);
  const [targetId, setTargetId] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (galleries.length === 0 || probes.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{t("search")}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{t("search_needs")}</p>
        </CardContent>
      </Card>
    );
  }

  async function runSearch() {
    if (!galleryId || !probeId) return;
    setBusy(true);
    setError(null);
    const { data, response, error: err } = await searchSearchPost({
      body: { gallery_run_id: galleryId, probe_run_id: probeId, top_k: topK },
    });
    setBusy(false);
    if (!response?.ok || !data) {
      const detail_ = (err as { detail?: unknown })?.detail;
      setError(typeof detail_ === "string" ? detail_ : `HTTP ${response?.status}`);
      setResult(null);
    } else {
      setResult(data);
    }
  }

  async function attachHit(hit: SearchResponse["hits"][number]) {
    if (!targetId || !galleryId) return;
    await proposeHypothesisCasesCaseIdTargetsTargetIdHypothesesPost({
      path: { case_id: detail.case_id, target_id: targetId },
      body: {
        kind: "appearance",
        run_id: galleryId,
        camera_id: hit.gallery_camera_id,
        track_id: hit.gallery_track_id,
        raw_score: hit.score,
        probability: hit.probability,
        stream: hit.stream,
      },
    });
    onChanged();
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("search")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-2">
            <Label>{t("gallery_run")}</Label>
            <Select
              value={galleryId}
              onValueChange={(v) => setGalleryId(v as string)}
            >
              <SelectTrigger className="w-64">
                <SelectValue placeholder={t("pick_run")} />
              </SelectTrigger>
              <SelectContent>
                {galleries.map((run) => (
                  <SelectItem key={run.run_id} value={run.run_id}>
                    {run.run_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>{t("probe_run")}</Label>
            <Select
              value={probeId}
              onValueChange={(v) => setProbeId(v as string)}
            >
              <SelectTrigger className="w-64">
                <SelectValue placeholder={t("pick_run")} />
              </SelectTrigger>
              <SelectContent>
                {probes.map((run) => (
                  <SelectItem key={run.run_id} value={run.run_id}>
                    {run.run_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="top-k">{t("top_k")}</Label>
            <Input
              id="top-k"
              type="number"
              min={1}
              max={200}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-24"
            />
          </div>
          <Button onClick={runSearch} disabled={busy || !galleryId || !probeId}>
            {t("run_search")}
          </Button>
        </div>
        {error && (
          <Alert variant="destructive" role="alert">
            <AlertTitle>{error}</AlertTitle>
          </Alert>
        )}
        {result && (
          <div className="space-y-3">
            {detail.targets.length > 0 && (
              <div className="flex items-end gap-2">
                <div className="space-y-2">
                  <Label>{t("attach_to")}</Label>
                  <Select
                    value={targetId}
                    onValueChange={(v) => setTargetId(v as string)}
                  >
                    <SelectTrigger className="w-64">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {detail.targets.map((target) => (
                        <SelectItem
                          key={target.target_id}
                          value={target.target_id}
                        >
                          {target.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
            {result.hits.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("no_hits")}</p>
            ) : (
              <div className="overflow-x-auto rounded-lg border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{t("camera")}</TableHead>
                      <TableHead>{t("track")}</TableHead>
                      <TableHead>{t("score")}</TableHead>
                      <TableHead>{t("probability")}</TableHead>
                      <TableHead />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.hits.map((hit, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-mono text-xs">
                          {hit.gallery_camera_id}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {hit.gallery_track_id}
                        </TableCell>
                        <TableCell>{hit.score.toFixed(3)}</TableCell>
                        <TableCell>
                          {hit.probability != null
                            ? `${(hit.probability * 100).toLocaleString(locale, { maximumFractionDigits: 1 })}%`
                            : t("uncalibrated")}
                        </TableCell>
                        <TableCell>
                          <Button
                            size="xs"
                            variant="outline"
                            disabled={!targetId}
                            onClick={() => attachHit(hit)}
                          >
                            {t("attach")}
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
