"use client";

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export type AppDataset = "cityflowv2" | "wildtrack";

interface DatasetState {
  selectedDataset: AppDataset;
  setSelectedDataset: (dataset: AppDataset) => void;
}

export const useDatasetStore = create<DatasetState>()(
  persist(
    (set) => ({
      selectedDataset: "cityflowv2",
      setSelectedDataset: (dataset) => set({ selectedDataset: dataset }),
    }),
    {
      name: "athar-dataset",
      storage: createJSONStorage(() => localStorage),
    }
  )
);