"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Loader2, Play } from "lucide-react";
import { ModelPicker } from "@/components/ModelPicker";
import { FusionWeightSliders } from "@/components/reid/FusionWeightSliders";
import { ImageUploader, type ReIDUploadImage } from "@/components/reid/ImageUploader";
import { RankedResults } from "@/components/reid/RankedResults";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { fusionReid, type FusionReIDResponsePayload } from "@/lib/api";
import { fetchModels, type ModelEntry } from "@/services/models";

function toPayload(images: ReIDUploadImage[]) {
  return images.map((image) => ({
    id: image.id,
    image_base64: image.imageBase64,
    metadata: { name: image.name },
  }));
}

function normaliseWeights(weights: Record<string, number>, ids: string[]) {
  const sum = ids.reduce((total, id) => total + Math.max(0, weights[id] ?? 0), 0);
  const equal = ids.length > 0 ? 1 / ids.length : 0;
  return Object.fromEntries(ids.map((id) => [id, sum > 0 ? Math.max(0, weights[id] ?? 0) / sum : equal]));
}

export default function FusionPage() {
  const [modelEntries, setModelEntries] = useState<ModelEntry[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>(["veri776_09v_v17_transreid", "veri776_clipsenet_v6"]);
  const [weights, setWeights] = useState<Record<string, number>>({ veri776_09v_v17_transreid: 0.3, veri776_clipsenet_v6: 0.7 });
  const [queries, setQueries] = useState<ReIDUploadImage[]>([]);
  const [gallery, setGallery] = useState<ReIDUploadImage[]>([]);
  const [rerank, setRerank] = useState(true);
  const [aqeK, setAqeK] = useState(3);
  const [result, setResult] = useState<FusionReIDResponsePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    void fetchModels({ task_type: "single_cam_reid", include_dead_ends: false }).then(setModelEntries).catch(() => setModelEntries([]));
  }, []);

  const selectedModels = useMemo(() => selectedIds.map((id) => modelEntries.find((model) => model.id === id)).filter(Boolean) as ModelEntry[], [modelEntries, selectedIds]);
  const normalizedWeights = useMemo(() => normaliseWeights(weights, selectedIds), [selectedIds, weights]);
  const canRun = selectedIds.length >= 2 && queries.length > 0 && gallery.length > 0 && !isRunning;

  function handleSelectedIds(nextIds: string[]) {
    setSelectedIds(nextIds);
    setWeights((current) => normaliseWeights(current, nextIds));
  }

  async function runFusion() {
    try {
      setIsRunning(true);
      setError(null);
      setResult(null);
      const response = await fusionReid({
        models: selectedIds.map((id) => ({ modelId: id, weight: normalizedWeights[id] ?? 0 })),
        queries: toPayload(queries),
        gallery: toPayload(gallery),
        topK: 10,
        rerank,
        aqeK,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Fusion request failed");
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
            <h1 className="text-2xl font-semibold tracking-normal">Fusion ReID</h1>
          </div>
          <Button type="button" onClick={() => void runFusion()} disabled={!canRun}>
            {isRunning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Run
          </Button>
        </div>

        <div className="grid gap-6 xl:grid-cols-[460px_1fr]">
          <aside className="space-y-6">
            <Card>
              <CardContent className="space-y-4 p-3">
                <h2 className="text-sm font-semibold">Models</h2>
                <ModelPicker
                  selectedId={null}
                  onSelect={() => undefined}
                  multiSelect
                  selectedIds={selectedIds}
                  onMultiSelect={handleSelectedIds}
                  taskType="single_cam_reid"
                  allowUnavailableSelection
                  compact
                />
              </CardContent>
            </Card>
            <Card>
              <CardContent className="space-y-5 p-4">
                <FusionWeightSliders models={selectedModels} weights={normalizedWeights} onChange={setWeights} />
                <div className="flex items-center gap-2">
                  <Checkbox id="fusion-rerank" checked={rerank} onCheckedChange={(value) => setRerank(Boolean(value))} />
                  <label htmlFor="fusion-rerank" className="text-sm font-medium">Rerank</label>
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
            {result?.warnings.map((warning) => <div key={warning} className="rounded-md border border-yellow-600/30 bg-yellow-600/10 p-3 text-sm">{warning}</div>)}
            <RankedResults title="Fused results" results={result?.results ?? []} queries={queries} gallery={gallery} />
            {result && result.components.length > 0 && (
              <div className="space-y-4">
                <h2 className="text-sm font-semibold">Individual model comparison</h2>
                {result.components.map((component) => (
                  <RankedResults
                    key={component.modelId}
                    title={`${component.modelId} (${(component.weight * 100).toFixed(0)}%)`}
                    results={component.results}
                    queries={queries}
                    gallery={gallery}
                  />
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}