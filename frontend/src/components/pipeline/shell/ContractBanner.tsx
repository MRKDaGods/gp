"use client";

import type { ReactNode } from "react";
import { HelpCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import type { StageNumber } from "@/types";

import { StageStatusBadge } from "../status/StageStatusBadge";
import type { StageStatus } from "../status/types";

export interface ContractChip {
  label: string;
  stage?: StageNumber;
  missing?: boolean;
  stale?: boolean;
  render?: ReactNode;
}

export interface BlockedRequirement {
  label: string;
  stage: StageNumber;
}

export interface ContractBannerProps {
  stage: StageNumber;
  title: string;
  needs?: ContractChip[];
  produces?: ContractChip[];
  status?: StageStatus;
  blockedBy?: BlockedRequirement | null;
  helpText?: string;
  onNavigateToStage?: (stage: StageNumber) => void;
  className?: string;
}

function ContractChipList({ label, chips }: { label: string; chips: ContractChip[] }) {
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1.5 text-xs">
      <span className="font-medium text-muted-foreground">{label}:</span>
      {chips.length === 0 ? (
        <Badge variant="outline" className="bg-muted/30 text-muted-foreground">
          -
        </Badge>
      ) : (
        chips.map((chip) => chip.render ? (
          <span key={`${label}-${chip.label}-${chip.stage ?? "none"}`} className="inline-flex max-w-full">
            {chip.render}
          </span>
        ) : (
          <Badge
            key={`${label}-${chip.label}-${chip.stage ?? "none"}`}
            variant="outline"
            className={cn(
              "max-w-[220px] truncate",
              chip.missing && "border-status-error/40 bg-status-error/10 text-status-error",
              chip.stale && "border-status-stale/40 bg-status-stale/10 text-status-stale"
            )}
          >
            {chip.label}
          </Badge>
        ))
      )}
    </div>
  );
}

export function ContractBanner({
  stage,
  title,
  needs = [],
  produces = [],
  status = "idle",
  blockedBy,
  helpText,
  onNavigateToStage,
  className,
}: ContractBannerProps) {
  return (
    <div className={cn("sticky top-0 z-20 border-b border-border/60 bg-card px-4 py-3 sm:px-6", className)}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="truncate text-base font-semibold sm:text-lg">Stage {stage} · {title}</h1>
            <StageStatusBadge status={status} />
            {blockedBy ? (
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-7 border-status-blocked/40 bg-status-blocked/10 px-2 text-xs text-status-blocked hover:bg-status-blocked/15 hover:text-status-blocked"
                onClick={() => onNavigateToStage?.(blockedBy.stage)}
              >
                Blocked: needs {blockedBy.label}
              </Button>
            ) : null}
          </div>
          <div className="flex min-w-0 flex-col gap-1 sm:flex-row sm:flex-wrap sm:gap-x-4">
            <ContractChipList label="Needs" chips={needs} />
            <ContractChipList label="Produces" chips={produces} />
          </div>
        </div>
        {helpText ? (
          <Popover>
            <PopoverTrigger asChild>
              <Button type="button" variant="ghost" size="icon" className="h-8 w-8 self-start lg:self-center" aria-label={`About Stage ${stage}`}>
                <HelpCircle className="h-4 w-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="max-w-[340px] text-sm leading-6">
              {helpText}
            </PopoverContent>
          </Popover>
        ) : null}
      </div>
    </div>
  );
}
