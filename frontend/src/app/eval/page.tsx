"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowLeft, Play } from "lucide-react";
import { EvalProgressPanel } from "@/components/eval/EvalProgressPanel";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { submitEval, type EvalType } from "@/lib/api";

const EVAL_TYPES: Array<{ id: EvalType; label: string }> = [
  { id: "veri776_transreid", label: "VeRi-776 TransReID" },
  { id: "veri776_clipsenet", label: "VeRi-776 CLIP-SENet" },
  { id: "cityflow_transreid", label: "CityFlowV2 TransReID" },
  { id: "veri776_14t_fusion", label: "VeRi-776 14t Fusion" },
];

export default function EvalPage() {
  const [evalType, setEvalType] = useState<EvalType>("veri776_transreid");
  const [overridesText, setOverridesText] = useState("{}");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit() {
    try {
      setIsSubmitting(true);
      setError(null);
      const parsed = JSON.parse(overridesText || "{}");
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Overrides must be a JSON object");
      }
      const response = await submitEval({ evalType, configOverrides: parsed as Record<string, unknown> });
      setJobId(response.jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Eval submission failed");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 lg:px-6">
        <div className="flex flex-col gap-4 border-b pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <Button asChild variant="ghost" size="sm" className="-ml-2 h-8 px-2">
              <Link href="/"><ArrowLeft className="mr-2 h-4 w-4" />Dashboard</Link>
            </Button>
            <h1 className="text-2xl font-semibold tracking-normal">Eval Runner</h1>
          </div>
          <Button type="button" onClick={() => void handleSubmit()} disabled={isSubmitting}>
            <Play className="mr-2 h-4 w-4" />
            Submit
          </Button>
        </div>

        <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
          <Card>
            <CardContent className="space-y-5 p-4">
              <div className="space-y-2">
                <Label htmlFor="eval-type">Eval type</Label>
                <Select value={evalType} onValueChange={(value) => setEvalType(value as EvalType)}>
                  <SelectTrigger id="eval-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {EVAL_TYPES.map((entry) => (
                      <SelectItem key={entry.id} value={entry.id}>{entry.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="eval-overrides">Config overrides</Label>
                <textarea
                  id="eval-overrides"
                  value={overridesText}
                  onChange={(event) => setOverridesText(event.target.value)}
                  spellCheck={false}
                  className="min-h-[220px] w-full resize-y rounded-md border bg-background px-3 py-2 font-mono text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                />
              </div>
              {error && <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
            </CardContent>
          </Card>

          <EvalProgressPanel jobId={jobId} />
        </div>
      </div>
    </main>
  );
}