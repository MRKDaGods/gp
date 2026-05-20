"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Info, RotateCcw } from "lucide-react";
import { ModelPicker } from "@/components/ModelPicker";
import { FusionWeightSliders } from "@/components/reid/FusionWeightSliders";
import { Button } from "@/components/ui/button";
import { fetchModels, type ModelEntry } from "@/services/models";
import { usePipelineStore, type PipelineFusionConfig, type PipelineFusionModel } from "@/store";

const CANONICAL_TRANSREID_ID = "veri776_09v_v17_transreid";
const CANONICAL_CLIPSENET_ID = "veri776_clipsenet_v6";
const MAX_FUSION_MODELS = 3;

const CANONICAL_FUSION: PipelineFusionConfig = {
  models: [
    { modelId: CANONICAL_TRANSREID_ID, weight: 0.3 },
    { modelId: CANONICAL_CLIPSENET_ID, weight: 0.7 },
  ],
  aqeK: 3,
  k1: 80,
  k2: 15,
  lambda: 0.2,
  rerank: true,
};

const EMPTY_FUSION: PipelineFusionConfig = {
  models: [],
  aqeK: 3,
  k1: 80,
  k2: 15,
  lambda: 0.2,
  rerank: true,
};

function toWeightRecord(models: PipelineFusionModel[]): Record<string, number> {
  return Object.fromEntries(models.map((model) => [model.modelId, model.weight]));
}

function normaliseModels(models: PipelineFusionModel[]): PipelineFusionModel[] {
  if (models.length === 0) return [];
  if (models.length === 1) return [{ modelId: models[0].modelId, weight: 1 }];

  const sum = models.reduce((total, model) => total + Math.max(0, model.weight), 0);
  const equal = 1 / models.length;
  return models.map((model) => ({
    modelId: model.modelId,
    weight: sum > 0 ? Math.max(0, model.weight) / sum : equal,
  }));
}

function buildModelsForIds(ids: string[], currentModels: PipelineFusionModel[]): PipelineFusionModel[] {
  const currentWeights = toWeightRecord(currentModels);
  const models = ids.map((modelId) => ({ modelId, weight: currentWeights[modelId] ?? 1 }));
  return normaliseModels(models);
}

export function FusionModelPanel() {
  const fusion = usePipelineStore((state) => state.fusion);
  const setFusionConfig = usePipelineStore((state) => state.setFusionConfig);
  const [registryModels, setRegistryModels] = useState<ModelEntry[]>([]);
  const [isLoadingRegistry, setIsLoadingRegistry] = useState(true);
  const [preseedWarning, setPreseedWarning] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    void fetchModels({ include_dead_ends: false })
      .then((models) => {
        if (!cancelled) setRegistryModels(models);
      })
      .catch(() => {
        if (!cancelled) setRegistryModels([]);
      })
      .finally(() => {
        if (!cancelled) setIsLoadingRegistry(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const hasCanonicalSeed = useMemo(() => {
    const ids = new Set(registryModels.map((model) => model.id));
    return ids.has(CANONICAL_TRANSREID_ID) && ids.has(CANONICAL_CLIPSENET_ID);
  }, [registryModels]);

  const applyCanonicalDefaults = useCallback(() => {
    if (hasCanonicalSeed) {
      setFusionConfig(CANONICAL_FUSION);
      setPreseedWarning(null);
      return;
    }

    setFusionConfig(EMPTY_FUSION);
    setPreseedWarning("14t pre-seed unavailable in current registry; pick models manually.");
  }, [hasCanonicalSeed, setFusionConfig]);

  useEffect(() => {
    if (fusion !== null || isLoadingRegistry) return;
    applyCanonicalDefaults();
  }, [applyCanonicalDefaults, fusion, isLoadingRegistry]);

  const activeFusion = fusion ?? EMPTY_FUSION;
  const selectedIds = activeFusion.models.map((model) => model.modelId);
  const weights = useMemo(() => toWeightRecord(activeFusion.models), [activeFusion.models]);
  const selectedModels = useMemo(
    () => selectedIds
      .map((id) => registryModels.find((model) => model.id === id))
      .filter(Boolean) as ModelEntry[],
    [registryModels, selectedIds]
  );

  const handleSelectedIds = useCallback(
    (nextIds: string[]) => {
      const limitedIds = nextIds.slice(0, MAX_FUSION_MODELS);
      setFusionConfig({
        ...activeFusion,
        models: buildModelsForIds(limitedIds, activeFusion.models),
      });
    },
    [activeFusion, setFusionConfig]
  );

  const handleWeightsChange = useCallback(
    (nextWeights: Record<string, number>) => {
      setFusionConfig({
        ...activeFusion,
        models: normaliseModels(
          selectedIds.map((modelId) => ({ modelId, weight: nextWeights[modelId] ?? 0 }))
        ),
      });
    },
    [activeFusion, selectedIds, setFusionConfig]
  );

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-yellow-600/30 bg-yellow-600/10 p-3 text-sm">
        <div className="flex items-start gap-2">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-yellow-700" />
          <p>14t canonical fusion was tuned on VeRi-776. Cross-domain use on CityFlowV2 may not match VeRi-776 metrics.</p>
        </div>
      </div>

      {preseedWarning ? (
        <div className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
          {preseedWarning}
        </div>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-muted-foreground">
          Pick 2 to 3 registry models. The config is staged locally until backend fusion wiring lands.
        </div>
        <Button type="button" variant="outline" size="sm" onClick={applyCanonicalDefaults}>
          <RotateCcw className="mr-2 h-4 w-4" />
          Use 14t canonical defaults
        </Button>
      </div>

      <ModelPicker
        selectedId={null}
        onSelect={() => undefined}
        multiSelect
        selectedIds={selectedIds}
        onMultiSelect={handleSelectedIds}
        allowUnavailableSelection
        compact
        respectDatasetFilter={false}
      />

      {selectedIds.length < 2 ? (
        <div className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
          Pick at least 2 models
        </div>
      ) : null}

      {selectedModels.length > 0 ? (
        <FusionWeightSliders models={selectedModels} weights={weights} onChange={handleWeightsChange} />
      ) : null}

      <div className="rounded-md border bg-muted/40 p-3 text-xs">
        <div className="mb-2 text-sm font-medium">Fusion config preview</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 font-mono sm:grid-cols-5">
          <div><span className="text-muted-foreground">aqe_k </span>{activeFusion.aqeK}</div>
          <div><span className="text-muted-foreground">k1 </span>{activeFusion.k1}</div>
          <div><span className="text-muted-foreground">k2 </span>{activeFusion.k2}</div>
          <div><span className="text-muted-foreground">λ </span>{activeFusion.lambda}</div>
          <div><span className="text-muted-foreground">rerank </span>{activeFusion.rerank ? "true" : "false"}</div>
        </div>
      </div>
    </div>
  );
}
