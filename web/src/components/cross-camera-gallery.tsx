"use client";

import { useTranslations } from "next-intl";
import { Badge } from "@/components/ui/badge";
import { identityColor } from "@/components/run-timeline";
import type {
  TimelineIdentityOut,
  TimelineMemberOut,
} from "@/lib/api";
import { API_URL } from "@/lib/client";

type Selection = { identity: TimelineIdentityOut; member: TimelineMemberOut };

function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// The headline visual: every cross-camera identity as a literal photo strip
// of its hops, split by class so "vehicles too" isn't buried in a filter.
export function CrossCameraGallery({
  runId,
  identities,
  selected,
  onSelect,
}: {
  runId: string;
  identities: TimelineIdentityOut[];
  selected: Selection | null;
  onSelect: (selection: Selection) => void;
}) {
  const t = useTranslations("timeline");
  const crossing = identities.filter((i) => i.cross_camera);
  const vehicles = crossing.filter((i) => i.entity_class !== "person");
  const persons = crossing.filter((i) => i.entity_class === "person");

  if (crossing.length === 0) return null;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <GalleryColumn
        title={t("gallery_vehicles")}
        empty={t("gallery_none_vehicles")}
        runId={runId}
        identities={vehicles}
        selected={selected}
        onSelect={onSelect}
      />
      <GalleryColumn
        title={t("gallery_persons")}
        empty={t("gallery_none_persons")}
        runId={runId}
        identities={persons}
        selected={selected}
        onSelect={onSelect}
      />
    </div>
  );
}

function GalleryColumn({
  title,
  empty,
  runId,
  identities,
  selected,
  onSelect,
}: {
  title: string;
  empty: string;
  runId: string;
  identities: TimelineIdentityOut[];
  selected: Selection | null;
  onSelect: (selection: Selection) => void;
}) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-muted-foreground">
        {title} · {identities.length}
      </h3>
      {identities.length === 0 ? (
        <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
          {empty}
        </p>
      ) : (
        <div className="space-y-2">
          {identities.map((identity) => (
            <IdentityCard
              key={identity.global_id}
              runId={runId}
              identity={identity}
              selected={selected}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function IdentityCard({
  runId,
  identity,
  selected,
  onSelect,
}: {
  runId: string;
  identity: TimelineIdentityOut;
  selected: Selection | null;
  onSelect: (selection: Selection) => void;
}) {
  const t = useTranslations("timeline");
  const entityLabel = (entityClass: string) => {
    const key = `class_${entityClass}`;
    return t.has(key) ? t(key) : entityClass;
  };
  const hops = identity.members
    .filter((m) => m.start_s !== null)
    .sort((a, b) => (a.start_s as number) - (b.start_s as number));

  return (
    <div className="rounded-md border p-2">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span
          className="inline-block size-2.5 rounded-full"
          style={{ backgroundColor: identityColor(identity.global_id) }}
        />
        <span className="font-mono text-xs">#{identity.global_id}</span>
        <Badge variant="outline" className="text-[10px]">
          {entityLabel(identity.entity_class)}
        </Badge>
        {identity.confidence !== null && (
          <span className="text-[10px] text-muted-foreground">
            {(identity.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <div dir="ltr" className="flex items-center gap-1 overflow-x-auto pb-1">
        {hops.map((member, i) => {
          const isSelected =
            selected?.member === member &&
            selected.identity.global_id === identity.global_id;
          return (
            <div key={`${member.camera_id}-${member.track_id}`} className="flex items-center gap-1">
              {i > 0 && (
                <span className="text-muted-foreground/50 shrink-0">→</span>
              )}
              <button
                type="button"
                onClick={() => onSelect({ identity, member })}
                className={`flex shrink-0 cursor-pointer flex-col items-center gap-1 rounded-md p-1.5 transition-all hover:scale-105 hover:bg-muted ${
                  isSelected ? "ring-2 ring-foreground" : ""
                }`}
              >
                {member.has_thumbnail ? (
                  // eslint-disable-next-line @next/next/no-img-element -- API-served evidence crop
                  <img
                    src={`${API_URL}/runs/${runId}/thumbs/${member.camera_id}/${member.track_id}`}
                    alt={`#${identity.global_id} ${member.camera_id}`}
                    className="size-20 rounded-md border object-cover"
                    onError={(e) => {
                      e.currentTarget.style.visibility = "hidden";
                    }}
                  />
                ) : (
                  <div
                    className="flex size-20 items-center justify-center rounded-md border bg-muted text-xs text-muted-foreground"
                    style={{ borderColor: identityColor(identity.global_id) }}
                  >
                    {entityLabel(identity.entity_class)}
                  </div>
                )}
                <span className="font-mono text-xs text-muted-foreground">
                  {member.camera_id}
                </span>
                <span className="text-xs text-muted-foreground">
                  {member.start_s !== null ? formatClock(member.start_s) : ""}
                </span>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
