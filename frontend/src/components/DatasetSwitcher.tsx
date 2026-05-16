"use client";

import { Database } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useDatasetStore, type AppDataset } from "@/lib/store";

const DATASETS: Array<{ id: AppDataset; label: string }> = [
  { id: "cityflowv2", label: "CityFlowV2" },
  { id: "wildtrack", label: "WILDTRACK" },
];

export function DatasetSwitcher() {
  const selectedDataset = useDatasetStore((state) => state.selectedDataset);
  const setSelectedDataset = useDatasetStore((state) => state.setSelectedDataset);

  return (
    <div className="fixed left-4 top-4 z-50 flex items-center gap-2 rounded-md border bg-background/95 px-2 py-1.5 shadow-sm backdrop-blur">
      <Database className="h-4 w-4 text-muted-foreground" />
      <Select value={selectedDataset} onValueChange={(value) => setSelectedDataset(value as AppDataset)}>
        <SelectTrigger className="h-8 w-[148px] border-0 px-2 shadow-none focus:ring-0 focus:ring-offset-0">
          <SelectValue />
        </SelectTrigger>
        <SelectContent align="start">
          {DATASETS.map((dataset) => (
            <SelectItem key={dataset.id} value={dataset.id}>{dataset.label}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}