"use client";

import { useCallback, useEffect } from "react";
import { Calendar, FolderOpen, MapPin } from "lucide-react";
import { format } from "date-fns";
import { create } from "zustand";

import { Button } from "@/components/ui/button";
import { Calendar as CalendarComponent } from "@/components/ui/calendar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getDatasets, getPipelineStatus, processDataset, type DatasetFolder } from "@/lib/api";
import { useSessionStore } from "@/store";

export const locationData = {
  governorates: [
    { id: "cairo", name: "Cairo", nameAr: "القاهرة" },
    { id: "giza", name: "Giza", nameAr: "الجيزة" },
    { id: "alexandria", name: "Alexandria", nameAr: "الإسكندرية" },
    { id: "aswan", name: "Aswan", nameAr: "أسوان" },
    { id: "luxor", name: "Luxor", nameAr: "الأقصر" },
  ],
  cities: {
    cairo: [
      { id: "downtown", name: "Downtown", nameAr: "وسط البلد" },
      { id: "heliopolis", name: "Heliopolis", nameAr: "مصر الجديدة" },
      { id: "maadi", name: "Maadi", nameAr: "المعادي" },
      { id: "nasr_city", name: "Nasr City", nameAr: "مدينة نصر" },
    ],
    giza: [
      { id: "dokki", name: "Dokki", nameAr: "الدقي" },
      { id: "mohandessin", name: "Mohandessin", nameAr: "المهندسين" },
      { id: "haram", name: "Haram", nameAr: "الهرم" },
    ],
    alexandria: [
      { id: "sidi_gaber", name: "Sidi Gaber", nameAr: "سيدي جابر" },
      { id: "stanley", name: "Stanley", nameAr: "ستانلي" },
    ],
  },
  zones: {
    downtown: [
      { id: "tahrir", name: "Tahrir Square", nameAr: "ميدان التحرير" },
      { id: "ramses", name: "Ramses", nameAr: "رمسيس" },
      { id: "attaba", name: "Attaba", nameAr: "العتبة" },
    ],
    heliopolis: [
      { id: "korba", name: "Korba", nameAr: "كوربة" },
      { id: "merghany", name: "Merghany", nameAr: "الميرغني" },
    ],
    maadi: [
      { id: "degla", name: "Degla", nameAr: "دجلة" },
      { id: "sarayat", name: "Sarayat", nameAr: "سرايات" },
    ],
  },
};

/** The sentinel dataset value meaning "search within my uploaded probe only". */
export const UPLOADED_DATASET = "__uploaded__";

export interface DatasetProcessProgress {
  progress: number;
  message: string;
}

interface InferenceSourceState {
  datasets: DatasetFolder[];
  selectedDataset: string;
  datasetsLoading: boolean;
  /** Name of the dataset currently being auto-processed (null when idle). */
  processingDataset: string | null;
  processProgress: DatasetProcessProgress | null;
  setDatasets: (datasets: DatasetFolder[]) => void;
  setSelectedDataset: (dataset: string) => void;
  setDatasetsLoading: (loading: boolean) => void;
  setProcessingDataset: (dataset: string | null) => void;
  setProcessProgress: (progress: DatasetProcessProgress | null) => void;
}

export const useInferenceSourceStore = create<InferenceSourceState>((set) => ({
  datasets: [],
  selectedDataset: UPLOADED_DATASET,
  datasetsLoading: true,
  processingDataset: null,
  processProgress: null,
  setDatasets: (datasets) => set({ datasets }),
  setSelectedDataset: (selectedDataset) => set({ selectedDataset }),
  setDatasetsLoading: (datasetsLoading) => set({ datasetsLoading }),
  setProcessingDataset: (processingDataset) => set({ processingDataset }),
  setProcessProgress: (processProgress) => set({ processProgress }),
}));

/** Precompute run id for a dataset folder, mirroring the backend convention. */
export function precomputeRunId(datasetName: string): string {
  return `dataset_precompute_${datasetName.trim().toLowerCase()}`;
}

export interface EnsureGalleryResult {
  galleryRunId: string | null;
  cameraCoordinates?: DatasetFolder["cameraCoordinates"];
}

/**
 * Ensure the chosen dataset has a ready search gallery, auto-processing it inline
 * if not. Returns the gallery run id to search within (or null for the uploaded
 * probe). Polls the stable precompute run until it completes.
 */
export function useEnsureGalleryReady() {
  const setProcessingDataset = useInferenceSourceStore((s) => s.setProcessingDataset);
  const setProcessProgress = useInferenceSourceStore((s) => s.setProcessProgress);

  return useCallback(
    async (datasetName: string): Promise<EnsureGalleryResult> => {
      // Uploaded probe: no external gallery needed.
      if (!datasetName || datasetName === UPLOADED_DATASET) {
        return { galleryRunId: null };
      }

      const readList = async (): Promise<DatasetFolder[]> => {
        const resp: any = await getDatasets();
        const data = resp?.data ?? resp;
        return Array.isArray(data) ? (data as DatasetFolder[]) : [];
      };

      let list = await readList();
      let ds = list.find((d) => d.name === datasetName) ?? null;

      // Already precomputed -> use it directly.
      if (ds?.hasGallery && ds.galleryRunId) {
        useInferenceSourceStore.getState().setDatasets(list);
        return { galleryRunId: ds.galleryRunId, cameraCoordinates: ds.cameraCoordinates };
      }

      // Not ready -> kick off precompute (idempotent on the backend: reprocess
      // overwrites the same dataset_precompute_<slug> dir) and poll to completion.
      const runId = precomputeRunId(datasetName);
      setProcessingDataset(datasetName);
      setProcessProgress({ progress: 0, message: `Processing ${datasetName}...` });
      try {
        if (!ds?.isProcessing) {
          await processDataset(datasetName);
        }

        // Poll the stable precompute run until it settles.
        // eslint-disable-next-line no-constant-condition
        while (true) {
          await new Promise((r) => setTimeout(r, 1500));
          let status: any = null;
          try {
            const resp = await getPipelineStatus(runId);
            status = resp?.data ?? resp;
          } catch {
            // transient - keep polling
          }
          if (status) {
            const state = String(status.status ?? "running");
            setProcessProgress({
              progress: Number(status.progress ?? 0),
              message: String(status.message ?? `Processing ${datasetName}...`),
            });
            if (state === "completed") break;
            if (state === "error") {
              throw new Error(String(status.error ?? status.message ?? "Dataset processing failed"));
            }
            if (state === "cancelled") {
              throw new Error("Dataset processing was cancelled");
            }
          }
        }

        // Refresh and resolve the freshly-built gallery.
        list = await readList();
        useInferenceSourceStore.getState().setDatasets(list);
        ds = list.find((d) => d.name === datasetName) ?? null;
        const galleryRunId = ds?.galleryRunId ?? runId;
        return { galleryRunId, cameraCoordinates: ds?.cameraCoordinates };
      } finally {
        setProcessingDataset(null);
        setProcessProgress(null);
      }
    },
    [setProcessingDataset, setProcessProgress]
  );
}

export function useInferenceDatasets() {
  const datasets = useInferenceSourceStore((state) => state.datasets);
  const selectedDataset = useInferenceSourceStore((state) => state.selectedDataset);
  const datasetsLoading = useInferenceSourceStore((state) => state.datasetsLoading);
  const setDatasets = useInferenceSourceStore((state) => state.setDatasets);
  const setDatasetsLoading = useInferenceSourceStore((state) => state.setDatasetsLoading);

  const fetchDatasets = useCallback(async () => {
    try {
      setDatasetsLoading(true);
      const response: any = await getDatasets();
      const data = response?.data ?? response;
      setDatasets(Array.isArray(data) ? data : []);
    } catch {
      setDatasets([]);
    } finally {
      setDatasetsLoading(false);
    }
  }, [setDatasets, setDatasetsLoading]);

  useEffect(() => {
    void fetchDatasets();
  }, [fetchDatasets]);

  return { datasets, selectedDataset, datasetsLoading, fetchDatasets };
}

export function InferenceSourceCard() {
  const { locationFilter, setLocationFilter, dateTimeRange, setDateTimeRange } = useSessionStore();

  const availableCities = locationFilter.governorate
    ? locationData.cities[locationFilter.governorate as keyof typeof locationData.cities] || []
    : [];
  const availableZones = locationFilter.city
    ? locationData.zones[locationFilter.city as keyof typeof locationData.zones] || []
    : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <FolderOpen className="h-5 w-5" />
          Source and Filters
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <section className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-medium"><MapPin className="h-4 w-4" />Location</div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Governorate</Label>
              <Select value={locationFilter.governorate} onValueChange={(value) => setLocationFilter({ governorate: value, city: undefined, zone: undefined })}>
                <SelectTrigger><SelectValue placeholder="Select governorate" /></SelectTrigger>
                <SelectContent>
                  {locationData.governorates.map((governorate) => (
                    <SelectItem key={governorate.id} value={governorate.id}>{governorate.name} <span className="text-xs text-muted-foreground">{governorate.nameAr}</span></SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>City</Label>
              <Select value={locationFilter.city} onValueChange={(value) => setLocationFilter({ city: value, zone: undefined })} disabled={!locationFilter.governorate}>
                <SelectTrigger><SelectValue placeholder="Select city" /></SelectTrigger>
                <SelectContent>
                  {availableCities.map((city) => (
                    <SelectItem key={city.id} value={city.id}>{city.name} <span className="text-xs text-muted-foreground">{city.nameAr}</span></SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Zone</Label>
              <Select value={locationFilter.zone} onValueChange={(value) => setLocationFilter({ zone: value })} disabled={!locationFilter.city}>
                <SelectTrigger><SelectValue placeholder="Select zone" /></SelectTrigger>
                <SelectContent>
                  {availableZones.map((zone) => (
                    <SelectItem key={zone.id} value={zone.id}>{zone.name} <span className="text-xs text-muted-foreground">{zone.nameAr}</span></SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>

        <section className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-medium"><Calendar className="h-4 w-4" />Date range</div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Start Date</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start text-left font-normal">
                    <Calendar className="mr-2 h-4 w-4" />
                    {dateTimeRange.start ? format(dateTimeRange.start, "PPP") : "Select start date"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <CalendarComponent mode="single" selected={dateTimeRange.start} onSelect={(date) => setDateTimeRange({ start: date })} initialFocus />
                </PopoverContent>
              </Popover>
            </div>

            <div className="space-y-2">
              <Label>End Date</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start text-left font-normal">
                    <Calendar className="mr-2 h-4 w-4" />
                    {dateTimeRange.end ? format(dateTimeRange.end, "PPP") : "Select end date"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <CalendarComponent mode="single" selected={dateTimeRange.end} onSelect={(date) => setDateTimeRange({ end: date })} initialFocus />
                </PopoverContent>
              </Popover>
            </div>
          </div>
        </section>
      </CardContent>
    </Card>
  );
}