"use client";

import { useCallback, useEffect } from "react";
import { Calendar, Camera, CheckCircle2, FolderOpen, Loader2, MapPin, Video } from "lucide-react";
import { format } from "date-fns";
import { create } from "zustand";

import { Button } from "@/components/ui/button";
import { Calendar as CalendarComponent } from "@/components/ui/calendar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { getDatasets, type DatasetFolder } from "@/lib/api";
import { cn } from "@/lib/utils";
import { usePipelineStore, useSessionStore, useVideoStore } from "@/store";

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

interface InferenceSourceState {
  datasets: DatasetFolder[];
  selectedDataset: string;
  datasetsLoading: boolean;
  setDatasets: (datasets: DatasetFolder[]) => void;
  setSelectedDataset: (dataset: string) => void;
  setDatasetsLoading: (loading: boolean) => void;
}

export const useInferenceSourceStore = create<InferenceSourceState>((set) => ({
  datasets: [],
  selectedDataset: "__uploaded__",
  datasetsLoading: true,
  setDatasets: (datasets) => set({ datasets }),
  setSelectedDataset: (selectedDataset) => set({ selectedDataset }),
  setDatasetsLoading: (datasetsLoading) => set({ datasetsLoading }),
}));

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
  const currentVideo = useVideoStore((state) => state.currentVideo);
  const selectedModelMeta = usePipelineStore((state) => state.selectedModelMeta);
  const { datasets, selectedDataset, datasetsLoading, fetchDatasets } = useInferenceDatasets();
  const setSelectedDataset = useInferenceSourceStore((state) => state.setSelectedDataset);
  const effectiveDatasetLabel = selectedModelMeta?.dataset ?? selectedDataset;

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
          <Label>Dataset source</Label>
          {selectedModelMeta ? (
            <div className="rounded-md border bg-muted/50 p-3">
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <Badge variant="secondary" className="uppercase">{effectiveDatasetLabel}</Badge>
                <span className="text-muted-foreground">Dataset is derived from the selected registry model.</span>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
              <button
                type="button"
                className={cn(
                  "rounded-md border-2 p-4 text-left transition-all hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  selectedDataset === "__uploaded__" ? "border-primary bg-primary/5 shadow-sm" : "border-muted hover:border-primary/50"
                )}
                onClick={() => setSelectedDataset("__uploaded__")}
                aria-pressed={selectedDataset === "__uploaded__"}
              >
                <div className="mb-2 flex items-center gap-2">
                  <Video className="h-5 w-5 text-primary" />
                  <span className="text-sm font-medium">Uploaded Video</span>
                </div>
                <p className="truncate text-xs text-muted-foreground">{currentVideo?.name ?? "No video uploaded"}</p>
              </button>

              {datasetsLoading ? (
                <div className="flex items-center justify-center rounded-md border-2 border-dashed border-muted p-4">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  <span className="text-sm text-muted-foreground">Loading datasets...</span>
                </div>
              ) : null}

              {datasets.map((dataset) => (
                <button
                  type="button"
                  key={dataset.name}
                  className={cn(
                    "rounded-md border-2 p-4 text-left transition-all hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                    selectedDataset === dataset.name ? "border-primary bg-primary/5 shadow-sm" : "border-muted hover:border-primary/50"
                  )}
                  onClick={() => setSelectedDataset(dataset.name)}
                  aria-pressed={selectedDataset === dataset.name}
                >
                  <div className="mb-2 flex items-center gap-2">
                    <FolderOpen className="h-5 w-5 text-amber-500" />
                    <span className="truncate text-sm font-medium">{dataset.name}</span>
                    {dataset.alreadyProcessed ? <CheckCircle2 className="ml-auto h-4 w-4 text-green-500" /> : null}
                  </div>
                  <p className="text-xs text-muted-foreground">{dataset.cameraCount} cameras, {dataset.videosFound} videos</p>
                  {dataset.alreadyProcessed ? (
                    <Badge variant="outline" className="mt-2 border-green-600 text-[10px] text-green-600">Pre-processed</Badge>
                  ) : null}
                </button>
              ))}

              {!datasetsLoading && datasets.length === 0 ? (
                <div className="rounded-md border-2 border-dashed border-muted p-4 text-center lg:col-span-2">
                  <p className="text-sm text-muted-foreground">No dataset folders found in dataset/</p>
                  <Button variant="ghost" size="sm" className="mt-1" onClick={() => void fetchDatasets()}>Retry</Button>
                </div>
              ) : null}
            </div>
          )}

          {!selectedModelMeta && selectedDataset !== "__uploaded__" ? (
            <div className="rounded-md border bg-muted/50 p-3">
              {(() => {
                const dataset = datasets.find((candidate) => candidate.name === selectedDataset);
                if (!dataset) return null;
                return (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between gap-3 text-sm">
                      <span className="font-medium">{dataset.name}</span>
                      {dataset.alreadyProcessed ? (
                        <Badge variant="outline" className="border-green-600 text-green-600">
                          <CheckCircle2 className="mr-1 h-3 w-3" />
                          Processed
                        </Badge>
                      ) : (
                        <Badge variant="secondary">Not processed</Badge>
                      )}
                    </div>
                    <div className="flex gap-4 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1"><Camera className="h-3 w-3" />{dataset.cameraCount} cameras</span>
                      <span className="flex items-center gap-1"><Video className="h-3 w-3" />{dataset.videosFound} videos</span>
                    </div>
                  </div>
                );
              })()}
            </div>
          ) : null}
        </section>

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