"use client";

import { cn } from "@/lib/utils";
import Image from "next/image";

interface LogoProps {
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
}

const sizes = {
  sm: 32,
  md: 48,
  lg: 64,
  xl: 120,
};

export function Logo({ size = "md", className }: LogoProps) {
  const px = sizes[size];
  return (
    <div className={cn("flex items-center", className)}>
      <Image
        src="/logo.png"
        alt="ATHAR"
        width={px}
        height={px}
        className="hidden object-contain dark:block"
        style={{ filter: "brightness(0) invert(1)" }}
        priority
      />
      <Image
        src="/logo.png"
        alt="ATHAR"
        width={px}
        height={px}
        className="block object-contain dark:hidden"
        priority
      />
    </div>
  );
}

export function LogoIcon({ size = 32, className }: { size?: number; className?: string }) {
  return (
    <>
      <Image
        src="/logo.png"
        alt="ATHAR"
        width={size}
        height={size}
        className={cn("hidden object-contain dark:block", className)}
        style={{ filter: "brightness(0) invert(1)" }}
      />
      <Image
        src="/logo.png"
        alt="ATHAR"
        width={size}
        height={size}
        className={cn("block object-contain dark:hidden", className)}
      />
    </>
  );
}
