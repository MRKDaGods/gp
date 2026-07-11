/**
 * Shared class metadata + tracking-domain helpers.
 *
 * The pipeline tracks either VEHICLES (CityFlowV2: car/bus/truck) or PEOPLE
 * (WILDTRACK: person). Every stage renderer used to hardcode the vehicle
 * classes, so the people path showed car icons, "Other" labels, and vehicle
 * copy. This module is the single source of truth for:
 *   - per-class label / icon / colour (keyed by COCO class id), and
 *   - the run's "domain" (vehicles vs people vs mixed) used for headings,
 *     empty-states, and singular/plural nouns.
 *
 * Prefer deriving the domain from the ACTUAL detected classes when they're
 * known (robust for re-opened runs and custom folders); fall back to the
 * selected dataset only before any detections exist.
 */
import {
  Bike,
  Bus,
  Car,
  PersonStanding,
  Truck,
  type LucideIcon,
} from "lucide-react";

import { getClassColor } from "@/lib/utils";

/** COCO class ids the pipeline can emit. */
export const PERSON_CLASS_IDS = new Set<number>([0]);
export const VEHICLE_CLASS_IDS = new Set<number>([1, 2, 3, 5, 7]); // bicycle, car, motorcycle, bus, truck

interface ClassMeta {
  label: string;
  icon: LucideIcon;
}

/** Per-class presentation. Colours come from utils.getClassColor (shared). */
const CLASS_META: Record<number, ClassMeta> = {
  0: { label: "Person", icon: PersonStanding },
  1: { label: "Bicycle", icon: Bike },
  2: { label: "Car", icon: Car },
  3: { label: "Motorcycle", icon: Bike },
  5: { label: "Bus", icon: Bus },
  7: { label: "Truck", icon: Truck },
};

/** Human label for a class id, falling back to the backend className, then a domain-neutral word. */
export function classLabelFor(classId: number | null | undefined, fallbackName?: string | null): string {
  if (classId != null && CLASS_META[classId]) return CLASS_META[classId].label;
  const name = (fallbackName ?? "").trim();
  if (name) return name.charAt(0).toUpperCase() + name.slice(1);
  return "Object";
}

/** Icon component for a class id (people get a pedestrian glyph, not a car). */
export function classIconFor(classId: number | null | undefined): LucideIcon {
  if (classId != null && CLASS_META[classId]) return CLASS_META[classId].icon;
  return Car;
}

/** Map a className string to a class id (for records that carry only a name). */
const NAME_TO_CLASS_ID: Record<string, number> = {
  person: 0, pedestrian: 0, people: 0,
  bicycle: 1, bike: 1, cyclist: 1,
  car: 2, sedan: 2, suv: 2, van: 2,
  motorcycle: 3, motorbike: 3,
  bus: 5,
  truck: 7, lorry: 7,
};

/** Resolve a className string to a COCO class id, or undefined when unknown. */
export function classIdFromName(name: string | null | undefined): number | undefined {
  return name ? NAME_TO_CLASS_ID[name.toLowerCase()] : undefined;
}

/** Icon component from a className string, falling back to a car glyph. */
export function classIconForName(name: string | null | undefined): LucideIcon {
  return classIconFor(classIdFromName(name));
}

/** Shared colour for a class id. */
export function classColorFor(classId: number | null | undefined): string {
  return getClassColor(classId ?? -1);
}

export type TrackingDomain = "vehicles" | "people" | "objects";

/** Map the app dataset selector to a domain (used before any detections exist). */
export function domainFromDataset(dataset: string | null | undefined): TrackingDomain {
  const d = (dataset ?? "").toLowerCase();
  if (/wildtrack|epfl|person|people|pedestrian/.test(d)) return "people";
  if (/cityflow|aic|veri|vehicle|car/.test(d)) return "vehicles";
  return "objects";
}

/** Derive the domain from the classes actually present. Null when none are known. */
export function domainFromClassIds(classIds: Iterable<number>): TrackingDomain | null {
  let hasPerson = false;
  let hasVehicle = false;
  let hasAny = false;
  for (const id of classIds) {
    hasAny = true;
    if (PERSON_CLASS_IDS.has(id)) hasPerson = true;
    else if (VEHICLE_CLASS_IDS.has(id)) hasVehicle = true;
  }
  if (!hasAny) return null;
  if (hasPerson && !hasVehicle) return "people";
  if (hasVehicle && !hasPerson) return "vehicles";
  if (hasPerson && hasVehicle) return "objects";
  return null; // only unknown class ids
}

/** Best-effort domain: prefer real detected classes, fall back to the dataset selector. */
export function resolveDomain(
  dataset: string | null | undefined,
  classIds?: Iterable<number>
): TrackingDomain {
  if (classIds) {
    const fromData = domainFromClassIds(classIds);
    if (fromData) return fromData;
  }
  return domainFromDataset(dataset);
}

/** Singular/plural noun for a domain, e.g. "vehicle" / "people" / "objects". */
export function domainNoun(
  domain: TrackingDomain,
  opts?: { plural?: boolean; cap?: boolean }
): string {
  const plural = opts?.plural ?? false;
  let word: string;
  if (domain === "people") word = plural ? "people" : "person";
  else if (domain === "vehicles") word = plural ? "vehicles" : "vehicle";
  else word = plural ? "objects" : "object";
  return opts?.cap ? word.charAt(0).toUpperCase() + word.slice(1) : word;
}

/** Panel heading like "Tracked Vehicles" / "Tracked People" / "Tracked Objects". */
export function trackedTitle(domain: TrackingDomain): string {
  if (domain === "people") return "Tracked People";
  if (domain === "vehicles") return "Tracked Vehicles";
  return "Tracked Objects";
}

/** A representative icon for a whole domain (empty-states, placeholders). */
export function domainIcon(domain: TrackingDomain): LucideIcon {
  if (domain === "people") return PersonStanding;
  return Car;
}
