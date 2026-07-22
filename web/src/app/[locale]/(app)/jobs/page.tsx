"use client";

import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";
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
    return <p className="text-zinc-500">…</p>;
  }
  return (
    <section className="space-y-4">
      <h1 className="text-xl font-bold">{t("title")}</h1>
      {jobs.length === 0 ? (
        <p className="text-zinc-500">{t("empty")}</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-zinc-400">
              <tr>
                {[t("job_id"), t("kind"), t("status"), t("executor"), ""].map(
                  (header, i) => (
                    <th key={i} className="px-4 py-2 text-start font-medium">
                      {header}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr
                  key={job.job_id}
                  className="border-t border-zinc-800 hover:bg-zinc-900/50"
                >
                  <td className="px-4 py-2 font-mono text-xs">{job.job_id}</td>
                  <td className="px-4 py-2">{job.kind}</td>
                  <td className="px-4 py-2">{job.status}</td>
                  <td className="px-4 py-2">{job.executor}</td>
                  <td className="px-4 py-2">
                    {ACTIVE.has(job.status ?? "") && (
                      <button
                        onClick={() => cancel(job.job_id!)}
                        className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 transition hover:bg-zinc-800"
                      >
                        {t("cancel")}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
