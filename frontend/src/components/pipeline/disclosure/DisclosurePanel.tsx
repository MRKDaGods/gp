"use client";

import { useId, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface DisclosurePanelProps {
  title?: string;
  description?: string;
  tier?: "advanced" | "debug";
  defaultOpen?: boolean;
  children?: ReactNode;
  className?: string;
}

export function DisclosurePanel({
  title,
  description,
  tier = "advanced",
  defaultOpen = false,
  children,
  className,
}: DisclosurePanelProps) {
  const [open, setOpen] = useState(defaultOpen);
  const panelId = useId();
  const resolvedTitle = title ?? (tier === "debug" ? "Debug" : "Advanced");

  return (
    <section
      className={cn(
        "rounded-md border bg-card",
        tier === "debug" && "border-dashed bg-muted/20",
        className
      )}
    >
      <Button
        type="button"
        variant="ghost"
        className="flex h-auto w-full items-start justify-between gap-3 rounded-md px-4 py-3 text-left"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="min-w-0 space-y-1">
          <span className="flex items-center gap-2 text-sm font-medium">
            {resolvedTitle}
            {tier === "debug" ? (
              <span className="rounded border border-dashed px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                Debug
              </span>
            ) : null}
          </span>
          {description ? <span className="block text-xs font-normal text-muted-foreground">{description}</span> : null}
        </span>
        <ChevronDown className={cn("mt-0.5 h-4 w-4 shrink-0 transition-transform", open && "rotate-180")} />
      </Button>
      {open ? (
        <div id={panelId} className="border-t px-4 py-4">
          {children}
        </div>
      ) : null}
    </section>
  );
}
