"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { listRunsRunsGet, type RunSummary } from "@/lib/api";
import "@/lib/client";

const STATUS_STYLES: Record<string, string> = {
  completed: "text-emerald-400",
  running: "text-sky-400",
  failed: "text-red-400",
  cancelled: "text-amber-400",
};

export default function RunsPage() {
  const t = useTranslations("runs");
  const [runs, setRuns] = useState<RunSummary[] | null>(null);

  useEffect(() => {
    listRunsRunsGet().then(({ data }) => setRuns(data ?? []));
  }, []);

  if (runs === null) {
    return <p className="text-zinc-500">…</p>;
  }
  return (
    <section className="space-y-4">
      <h1 className="text-xl font-bold">{t("title")}</h1>
      {runs.length === 0 ? (
        <p className="text-zinc-500">{t("empty")}</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-start text-zinc-400">
              <tr>
                {[
                  t("run_id"),
                  t("role"),
                  t("status"),
                  t("profile"),
                  t("artifacts"),
                  t("created"),
                ].map((header) => (
                  <th key={header} className="px-4 py-2 text-start font-medium">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.run_id}
                  className="border-t border-zinc-800 hover:bg-zinc-900/50"
                >
                  <td className="px-4 py-2 font-mono text-xs">{run.run_id}</td>
                  <td className="px-4 py-2">{run.role}</td>
                  <td
                    className={`px-4 py-2 ${STATUS_STYLES[run.status] ?? "text-zinc-300"}`}
                  >
                    {run.status}
                  </td>
                  <td className="px-4 py-2">{run.profile_name}</td>
                  <td className="px-4 py-2">{run.num_artifacts}</td>
                  <td className="px-4 py-2 text-zinc-400">
                    {new Date(run.created_at).toLocaleString()}
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
