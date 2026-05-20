"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowLeft, Loader2, Play } from "lucide-react";
import { ModelPicker } from "@/components/ModelPicker";
import { ImageUploader, type ReIDUploadImage } from "@/components/reid/ImageUploader";
import { RankedResults } from "@/components/reid/RankedResults";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { singleCamReid, type SingleCamReIDResponsePayload } from "@/lib/api";

function toPayload(images: ReIDUploadImage[]) {
  return images.map((image) => ({
    id: image.id,
    image_base64: image.imageBase64,
    metadata: { name: image.name },
  }));
}

export default function ReIDPage() {
  const [selectedModelId, setSelectedModelId] = useState<string | null>("veri776_09v_v17_transreid");
  const [queries, setQueries] = useState<ReIDUploadImage[]>([]);
  const [gallery, setGallery] = useState<ReIDUploadImage[]>([]);
  const [rerank, setRerank] = useState(false);
  const [aqeK, setAqeK] = useState(0);
  const [result, setResult] = useState<SingleCamReIDResponsePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const canRun = useMemo(() => Boolean(selectedModelId && queries.length > 0 && gallery.length > 0 && !isRunning), [gallery.length, isRunning, queries.length, selectedModelId]);

  async function runReid() {
    if (!selectedModelId) return;
    try {
      setIsRunning(true);
      setError(null);
      setResult(null);
      const response = await singleCamReid({
        modelId: selectedModelId,
        queries: toPayload(queries),
        gallery: toPayload(gallery),
        topK: 10,
        rerank,
        aqeK,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "ReID request failed");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 lg:px-6">
        <div className="flex flex-col gap-4 border-b pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1">
            <Button asChild variant="ghost" size="sm" className="-ml-2 h-8 px-2">
              <Link href="/"><ArrowLeft className="mr-2 h-4 w-4" />Dashboard</Link>
            </Button>
            <h1 className="text-2xl font-semibold tracking-normal">Single-Cam ReID</h1>
          </div>
          <Button type="button" onClick={() => void runReid()} disabled={!canRun}>
            {isRunning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Run
          </Button>
        </div>

        <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
          <aside className="space-y-6">
            <Card>
              <CardContent className="space-y-4 p-3">
                <h2 className="text-sm font-semibold">Model</h2>
                <ModelPicker
                  selectedId={selectedModelId}
                  onSelect={setSelectedModelId}
                  taskType="single_cam_reid"
                  allowUnavailableSelection
                  compact
                />
              </CardContent>
            </Card>
            <Card>
              <CardContent className="space-y-5 p-4">
                <div className="flex items-center gap-2">
                  <Checkbox id="rerank" checked={rerank} onCheckedChange={(value) => setRerank(Boolean(value))} />
                  <label htmlFor="rerank" className="text-sm font-medium">Rerank</label>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">AQE K</span>
                    <span className="tabular-nums text-muted-foreground">{aqeK}</span>
                  </div>
                  <Slider min={0} max={20} step={1} value={[aqeK]} onValueChange={([value]) => setAqeK(value)} />
                </div>
              </CardContent>
            </Card>
          </aside>

          <section className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              <Card><CardContent className="p-4"><ImageUploader label="Query images" images={queries} onChange={setQueries} /></CardContent></Card>
              <Card><CardContent className="p-4"><ImageUploader label="Gallery images" images={gallery} onChange={setGallery} maxFiles={500} /></CardContent></Card>
            </div>
            {error && <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
            <RankedResults results={result?.results ?? []} queries={queries} gallery={gallery} />
          </section>
        </div>
      </div>
    </main>
  );
}