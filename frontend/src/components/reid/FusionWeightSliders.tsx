"use client";

import type { ModelEntry } from "@/services/models";
import { Slider } from "@/components/ui/slider";

interface FusionWeightSlidersProps {
  models: ModelEntry[];
  weights: Record<string, number>;
  onChange: (weights: Record<string, number>) => void;
}

function normalise(weights: Record<string, number>, ids: string[]): Record<string, number> {
  const sum = ids.reduce((total, id) => total + Math.max(0, weights[id] ?? 0), 0);
  if (sum <= 0) {
    const equal = ids.length > 0 ? 1 / ids.length : 0;
    return Object.fromEntries(ids.map((id) => [id, equal]));
  }
  return Object.fromEntries(ids.map((id) => [id, Math.max(0, weights[id] ?? 0) / sum]));
}

export function FusionWeightSliders({ models, weights, onChange }: FusionWeightSlidersProps) {
  const ids = models.map((model) => model.id);
  const normalized = normalise(weights, ids);

  function setWeight(modelId: string, nextWeight: number) {
    const clamped = Math.min(1, Math.max(0, nextWeight));
    const others = ids.filter((id) => id !== modelId);
    const remaining = 1 - clamped;
    const otherSum = others.reduce((total, id) => total + (normalized[id] ?? 0), 0);
    const next: Record<string, number> = { [modelId]: clamped };
    for (const id of others) {
      next[id] = otherSum > 0 ? ((normalized[id] ?? 0) / otherSum) * remaining : remaining / Math.max(1, others.length);
    }
    onChange(next);
  }

  if (models.length === 0) {
    return <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">Select two or more models.</div>;
  }

  return (
    <div className="space-y-4">
      {models.map((model) => (
        <div key={model.id} className="space-y-2 rounded-md border p-3">
          <div className="flex items-center justify-between gap-3">
            <span className="truncate text-sm font-medium">{model.name}</span>
            <span className="text-sm tabular-nums">{((normalized[model.id] ?? 0) * 100).toFixed(0)}%</span>
          </div>
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={[normalized[model.id] ?? 0]}
            onValueChange={([value]) => setWeight(model.id, value)}
          />
        </div>
      ))}
    </div>
  );
}