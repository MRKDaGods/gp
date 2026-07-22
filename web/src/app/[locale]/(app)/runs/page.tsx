"use client";

import { useLocale, useTranslations } from "next-intl";
import Link from "next/link";
import { useEffect, useState } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { listRunsRunsGet, type RunSummary } from "@/lib/api";
import "@/lib/client";

export default function RunsPage() {
  const t = useTranslations("runs");
  const locale = useLocale();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);

  useEffect(() => {
    listRunsRunsGet().then(({ data }) => setRuns(data ?? []));
  }, []);

  if (runs === null) {
    return <Skeleton className="h-32 w-full" />;
  }
  return (
    <section className="space-y-4">
      <h1 className="text-xl font-bold">{t("title")}</h1>
      {runs.length === 0 ? (
        <p className="text-muted-foreground">{t("empty")}</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("run_id")}</TableHead>
                <TableHead>{t("role")}</TableHead>
                <TableHead>{t("status")}</TableHead>
                <TableHead>{t("profile")}</TableHead>
                <TableHead>{t("artifacts")}</TableHead>
                <TableHead>{t("created")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs.map((run) => (
                <TableRow key={run.run_id}>
                  <TableCell className="font-mono text-xs">
                    <Link
                      href={`/${locale}/runs/${run.run_id}`}
                      className="hover:underline"
                    >
                      {run.run_id}
                    </Link>
                  </TableCell>
                  <TableCell>{run.role}</TableCell>
                  <TableCell>
                    <StatusBadge status={run.status} />
                  </TableCell>
                  <TableCell>{run.profile_name}</TableCell>
                  <TableCell>{run.num_artifacts}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {new Date(run.created_at).toLocaleString(locale)}
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
