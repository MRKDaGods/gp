"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "value" | "onChange"> {
  value?: number[];
  onValueChange?: (value: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
  tone?: "default" | "muted";
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, value, onValueChange, min = 0, max = 100, step = 1, disabled, tone = "default", ...props }, ref) => {
    const current = value?.[0] ?? min;
    const pct = max > min ? ((current - min) / (max - min)) * 100 : 0;

    return (
      <div className={cn("relative flex w-full touch-none select-none items-center", className)}>
        <input
          ref={ref}
          type="range"
          min={min}
          max={max}
          step={step}
          value={current}
          disabled={disabled}
          onChange={(e) => onValueChange?.([Number(e.target.value)])}
          className="slider-input absolute inset-0 z-10 h-full w-full cursor-pointer opacity-0 disabled:cursor-default"
          {...props}
        />
        <div className={cn(
          "relative h-2 w-full overflow-hidden rounded-full",
          tone === "muted" ? "bg-muted/60" : "bg-secondary"
        )}>
          <div
            className={cn(
              "absolute h-full rounded-full transition-[width] duration-75",
              tone === "muted" ? "bg-foreground/22" : "bg-primary"
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div
          className="pointer-events-none absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-5 w-5 rounded-full border-2 border-primary bg-background shadow-sm transition-[left] duration-75 disabled:opacity-50"
          style={{ left: `${pct}%` }}
        />
      </div>
    );
  }
);
Slider.displayName = "Slider"

export { Slider }
