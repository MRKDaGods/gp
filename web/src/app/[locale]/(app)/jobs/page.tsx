"use client";

import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
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
  cancelJobJobsJobIdCancelPost,
  listJobsJobsGet,
  type Job,
} from "@/lib/api";
import "@/lib/client";

const ACTIVE = new Set(["queued", "claimed", "running"]);

export default function JobsPage() {
  const t = useTranslations("jobs");
  const [jobs, setJobs] = useState<Job[] | null>(null);

  const refresh = useCallback(() => {
    listJobsJobsGet().then(({ data }) => setJobs(data ?? []));
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  async function cancel(jobId: string) {
    await cancelJobJobsJobIdCancelPost({ path: { job_id: jobId } });
    refresh();
  }

  if (jobs === null) {
    return <Skeleton className="h-32 w-full" />;
  }
  return (
    <section className="space-y-4">
      <h1 className="text-xl font-bold">{t("title")}</h1>
      {jobs.length === 0 ? (
        <p className="text-muted-foreground">{t("empty")}</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("job_id")}</TableHead>
                <TableHead>{t("kind")}</TableHead>
                <TableHead>{t("status")}</TableHead>
                <TableHead>{t("executor")}</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((job) => (
                <TableRow key={job.job_id}>
                  <TableCell className="font-mono text-xs">
                    {job.job_id}
                  </TableCell>
                  <TableCell>{job.kind}</TableCell>
                  <TableCell>
                    <StatusBadge status={job.status ?? "queued"} />
                  </TableCell>
                  <TableCell>{job.executor}</TableCell>
                  <TableCell>
                    {ACTIVE.has(job.status ?? "") && (
                      <Button
                        variant="outline"
                        size="xs"
                        onClick={() => cancel(job.job_id!)}
                      >
                        {t("cancel")}
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </section>
  );
}
