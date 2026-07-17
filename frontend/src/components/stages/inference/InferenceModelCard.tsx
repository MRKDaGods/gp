"use client";

import { useCallback } from "react";
import { Cpu } from "lucide-react";

import { ModelPicker } from "@/components/ModelPicker";
import { FusionModelPanel } from "@/components/stages/fusion-model-panel";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fetchModel, type ModelEntry, type ModelMetric } from "@/services/models";
import { usePipelineStore } from "@/store";

const HEADLINE_METRIC_PRIORITY = ["IDF1", "mAP", "R1"];

function metricLabel(metric: ModelMetric): string {
  return metric.name.replace(/^mtmc_/i, "").replace(/_/g, " ").toUpperCase();
}

function getHeadlineMetric(model: ModelEntry | null): ModelMetric | null {
  if (!model) return null;

  for (const metricName of HEADLINE_METRIC_PRIORITY) {
    const metric = model.metrics.find(
      (candidate) => candidate.verified && metricLabel(candidate).toLowerCase() === metricName.toLowerCase()
    );
    if (metric) return metric;
  }

  return model.metrics.find((candidate) => candidate.verified) ?? model.metrics[0] ?? null;
}

function formatMetric(metric: ModelMetric | null): string | null {
  if (!metric) return null;
  const value = metric.value >= 10 ? metric.value.toFixed(2) : metric.value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return `${metricLabel(metric)} ${value}`;
}

function SummaryChip({ modelMode, selectedModelMeta, fusion }: {
  modelMode: "single" | "fusion";
  selectedModelMeta: ModelEntry | null;
  fusion: ReturnType<typeof usePipelineStore.getState>["fusion"];
}) {
  if (modelMode === "fusion") {
    const modelCount = fusion?.models.length ?? 0;
    const summary = modelCount > 0
      ? `${modelCount} models * ${fusion?.models.map((model) => `${model.modelId} ${(model.weight * 100).toFixed(0)}%`).join(" + ")}`
      : "Pick 2 to 3 models";

    return <Badge variant="secondary" className="max-w-full truncate">Fusion * {summary}</Badge>;
  }

  const metric = formatMetric(getHeadlineMetric(selectedModelMeta));
  const label = selectedModelMeta
    ? `${selectedModelMeta.id}${metric ? ` * ${metric}` : ""}`
    : "legacy config * configs/default.yaml";

  return <Badge variant="secondary" className="max-w-full truncate">{label}</Badge>;
}

export function InferenceModelCard() {
  const modelMode = usePipelineStore((state) => state.modelMode);
  const setModelMode = usePipelineStore((state) => state.setModelMode);
  const selectedModelId = usePipelineStore((state) => state.selectedModelId);
  const selectedModelMeta = usePipelineStore((state) => state.selectedModelMeta);
  const setSelectedModel = usePipelineStore((state) => state.setSelectedModel);
  const clearSelectedModel = usePipelineStore((state) => state.clearSelectedModel);
  const fusion = usePipelineStore((state) => state.fusion);

  const handleSingleModelSelect = useCallback(
    (modelId: string | null) => {
      if (!modelId) {
        clearSelectedModel();
        return;
      }

      void fetchModel(modelId)
        .then((model) => setSelectedModel(model.id, model))
        .catch(() => {
          clearSelectedModel();
        });
    },
    [clearSelectedModel, setSelectedModel]
  );

  const handleSingleModelChange = useCallback(
    (model: ModelEntry | null) => {
      if (model) {
        setSelectedModel(model.id, model);
      } else if (!selectedModelId) {
        clearSelectedModel();
      }
    },
    [clearSelectedModel, selectedModelId, setSelectedModel]
  );

  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="h-5 w-5" />
            Model
          </CardTitle>
          <Tabs value={modelMode} onValueChange={(value) => setModelMode(value as "single" | "fusion")}>
            <TabsList aria-label="Inference model mode">
              <TabsTrigger value="single">Single</TabsTrigger>
              <TabsTrigger value="fusion">Fusion</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        <SummaryChip modelMode={modelMode} selectedModelMeta={selectedModelMeta} fusion={fusion} />
      </CardHeader>
      <CardContent className="p-4 pt-0">
        {modelMode === "single" ? (
          <ModelPicker
            selectedId={selectedModelId}
            onSelect={handleSingleModelSelect}
            onModelChange={handleSingleModelChange}
            defaultReadyOnly
            autoSelectReady
          />
        ) : (
          <FusionModelPanel />
        )}
      </CardContent>
    </Card>
  );
}