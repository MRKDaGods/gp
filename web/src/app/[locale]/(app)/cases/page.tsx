"use client";

import { useLocale, useTranslations } from "next-intl";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  createCaseCasesPost,
  listCasesCasesGet,
  type CaseSummary,
} from "@/lib/api";
import "@/lib/client";

export default function CasesPage() {
  const t = useTranslations("cases");
  const locale = useLocale();
  const router = useRouter();
  const [cases, setCases] = useState<CaseSummary[] | null>(null);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    listCasesCasesGet().then(({ data }) => setCases(data ?? []));
  }, []);

  async function onCreate(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    const { data, response } = await createCaseCasesPost({ body: { title } });
    setBusy(false);
    if (response?.ok && data) {
      router.push(`/${locale}/cases/${data.case_id}`);
    }
  }

  if (cases === null) {
    return <Skeleton className="h-32 w-full" />;
  }
  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">{t("title")}</h1>
        <Dialog>
          <DialogTrigger render={<Button size="sm" />}>
            {t("new_case")}
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("new_case")}</DialogTitle>
            </DialogHeader>
            <form onSubmit={onCreate} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="case-title">{t("case_title")}</Label>
                <Input
                  id="case-title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                  minLength={1}
                  maxLength={256}
                />
              </div>
              <Button type="submit" disabled={busy} className="w-full">
                {t("create")}
              </Button>
            </form>
          </DialogContent>
        </Dialog>
      </div>
      {cases.length === 0 ? (
        <p className="text-muted-foreground">{t("empty")}</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("case_title")}</TableHead>
                <TableHead>{t("status")}</TableHead>
                <TableHead>{t("owner")}</TableHead>
                <TableHead>{t("runs")}</TableHead>
                <TableHead>{t("targets")}</TableHead>
                <TableHead>{t("updated")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {cases.map((c) => (
                <TableRow key={c.case_id}>
                  <TableCell>
                    <Link
                      href={`/${locale}/cases/${c.case_id}`}
                      className="font-medium hover:underline"
                    >
                      {c.title}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={c.status} />
                  </TableCell>
                  <TableCell>{c.owner}</TableCell>
                  <TableCell>{c.num_runs}</TableCell>
                  <TableCell>{c.num_targets}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {new Date(c.updated_at).toLocaleString(locale)}
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
