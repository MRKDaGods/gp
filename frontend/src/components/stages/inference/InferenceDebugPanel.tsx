"use client";

import { Database } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { usePipelineStore } from "@/store";
import type { RunModelMetadata } from "@/types";

interface InferenceDebugPanelProps {
  runModelMetadata?: RunModelMetadata | null;
}

function JsonDetails({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="rounded-md border bg-muted/40 p-3">
      <summary className="cursor-pointer text-sm font-medium">{title}</summary>
      <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-background p-3 font-mono text-[11px]">
        {JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}

export function InferenceDebugPanel({ runModelMetadata }: InferenceDebugPanelProps) {
  const modelMode = usePipelineStore((state) => state.modelMode);
  const selectedModelMeta = usePipelineStore((state) => state.selectedModelMeta);
  const fusion = usePipelineStore((state) => state.fusion);

  const appliedOverrides = runModelMetadata?.appliedOverrides ?? selectedModelMeta?.model_overrides ?? [];
  const effectiveConfig = {
    mode: modelMode,
    model_id: runModelMetadata?.modelId ?? selectedModelMeta?.id ?? null,
    resolved_config: runModelMetadata?.resolvedConfig ?? selectedModelMeta?.pipeline_config ?? "configs/default.yaml",
    applied_overrides: appliedOverrides,
    fusion: runModelMetadata?.fusion_resolved ?? fusion ?? null,
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Database className="h-5 w-5" />
          Inference Debug
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-xs">
        <JsonDetails title="Effective config" value={effectiveConfig} />

        {runModelMetadata?.fusion_resolved ? (
          <div className="space-y-3 rounded-md border bg-muted/40 p-3">
            <div className="text-sm font-medium">Resolved fusion</div>
            <div className="space-y-1.5">
              {runModelMetadata.fusion_resolved.models.map((model) => (
                <div
                  key={model.model_id}
                  className={cn(
                    "grid grid-cols-[1fr_auto_auto] items-center gap-3 rounded-md px-2 py-1",
                    model.primary && "border border-primary/30 bg-primary/10"
                  )}
                >
                  <span className="truncate font-mono">{model.model_id}</span>
                  {model.primary ? <Badge variant="secondary">Primary</Badge> : <span className="text-muted-foreground">Secondary</span>}
                  <span className="font-mono tabular-nums">{(model.weight * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 rounded-md border bg-background p-3 font-mono sm:grid-cols-5">
              <div><span className="text-muted-foreground">aqe_k </span>{runModelMetadata.fusion_resolved.aqe_k}</div>
              <div><span className="text-muted-foreground">k1 </span>{runModelMetadata.fusion_resolved.k1}</div>
              <div><span className="text-muted-foreground">k2 </span>{runModelMetadata.fusion_resolved.k2}</div>
              <div><span className="text-muted-foreground">lambda </span>{runModelMetadata.fusion_resolved.lambda}</div>
              <div><span className="text-muted-foreground">rerank </span>{runModelMetadata.fusion_resolved.rerank ? "true" : "false"}</div>
            </div>
          </div>
        ) : null}

        <div className="rounded-md border bg-muted/40 p-3">
          <div className="mb-2 text-sm font-medium">Active Pipeline Parameters</div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
            {modelMode === "fusion" && fusion ? (
              <>
                <div className="text-muted-foreground">Mode</div>
                <div className="font-mono">Fusion ({fusion.models.length} models)</div>
                {fusion.models.map((model) => (
                  <div key={`active-${model.modelId}`} className="contents">
                    <div className="truncate text-muted-foreground">{model.modelId}</div>
                    <div className="font-mono tabular-nums">weight {(model.weight * 100).toFixed(0)}%</div>
                  </div>
                ))}
                <div className="text-muted-foreground">Fusion hyperparams</div>
                <div className="font-mono">aqe_k={fusion.aqeK} · k1={fusion.k1} · k2={fusion.k2} · lambda={fusion.lambda} · rerank={fusion.rerank ? "true" : "false"}</div>
              </>
            ) : null}
            <div className="text-muted-foreground">Detector</div>
            <div className="font-mono">YOLOv26 · conf 0.25 · IoU 0.65</div>
            <div className="text-muted-foreground">Tracker</div>
            <div className="font-mono">DeepOCSort · max_age 30</div>
            <div className="text-muted-foreground">ReID backbone</div>
            <div className="font-mono">TransReID ViT-Base · 768D to 280D PCA</div>
            <div className="text-muted-foreground">Samples / tracklet</div>
            <div className="font-mono">32 · flip_augment yes · cam_BN yes</div>
            <div className="text-muted-foreground">Quality filter</div>
            <div className="font-mono">Laplacian var &gt;= 15 · temp 3.0</div>
            <div className="text-muted-foreground">Matching</div>
            <div className="font-mono">FAISS IndexFlatIP · threshold 0.60</div>
            <div className="text-muted-foreground">Solver</div>
            <div className="font-mono">conflict_free_cc · AQE k=5 alpha=5.0</div>
            <div className="text-muted-foreground">FIC whitening</div>
            <div className="font-mono">reg 0.3 · gallery_expansion rounds=2</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}