"use client";

import { cn } from "@/lib/utils";
import Image from "next/image";

interface LogoProps {
  size?: "sm" | "md" | "lg" | "xl";
  showText?: boolean;
  className?: string;
}

const sizes = {
  sm: { icon: 32, text: "text-sm" },
  md: { icon: 48, text: "text-base" },
  lg: { icon: 64, text: "text-xl" },
  xl: { icon: 120, text: "text-3xl" },
};

export function Logo({ size = "md", showText = true, className }: LogoProps) {
  const s = sizes[size];

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <Image
        src="/logo.png"
        alt="ATHAR Logo"
        width={s.icon}
        height={s.icon}
        className="object-contain"
        priority
      />
      {showText && size !== "sm" && (
        <div className="flex flex-col">
          <span className={cn("font-bold tracking-tight text-foreground", s.text)}>
            ATHAR
          </span>
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
            Vehicle Tracking System
          </span>
        </div>
      )}
    </div>
  );
}

export function LogoIcon({ size = 32, className }: { size?: number; className?: string }) {
  return (
    <Image
      src="/logo.png"
      alt="ATHAR"
      width={size}
      height={size}
      className={cn("object-contain", className)}
    />
  );
}
