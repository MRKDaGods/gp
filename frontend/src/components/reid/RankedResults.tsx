"use client";

import type { ReIDQueryResult } from "@/lib/api";
import type { ReIDUploadImage } from "@/components/reid/ImageUploader";

interface RankedResultsProps {
  title?: string;
  results: ReIDQueryResult[];
  queries: ReIDUploadImage[];
  gallery: ReIDUploadImage[];
}

export function RankedResults({ title = "Ranked results", results, queries, gallery }: RankedResultsProps) {
  const queryById = new Map(queries.map((image) => [image.id, image]));
  const galleryById = new Map(gallery.map((image) => [image.id, image]));

  if (results.length === 0) {
    return <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">No ranked results yet.</div>;
  }

  return (
    <section className="space-y-4">
      <h2 className="text-sm font-semibold">{title}</h2>
      <div className="space-y-4">
        {results.map((result) => {
          const query = queryById.get(result.queryId);
          return (
            <div key={result.queryId} className="grid gap-3 rounded-md border p-3 lg:grid-cols-[120px_1fr]">
              <div className="space-y-2">
                <div className="aspect-square overflow-hidden rounded-md bg-muted">
                  {query && <img src={query.previewUrl} alt={query.name} className="h-full w-full object-cover" />}
                </div>
                <p className="truncate text-xs text-muted-foreground">{query?.name ?? result.queryId}</p>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-5">
                {result.matches.map((match) => {
                  const image = galleryById.get(match.galleryId);
                  return (
                    <div key={`${result.queryId}-${match.galleryId}`} className="overflow-hidden rounded-md border bg-card">
                      <div className="aspect-square bg-muted">
                        {image && <img src={image.previewUrl} alt={image.name} className="h-full w-full object-cover" />}
                      </div>
                      <div className="space-y-1 p-2">
                        <div className="flex items-center justify-between gap-2 text-xs font-medium">
                          <span>#{match.rank}</span>
                          <span>{match.score.toFixed(4)}</span>
                        </div>
                        <p className="truncate text-[11px] text-muted-foreground">{image?.name ?? match.galleryId}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}