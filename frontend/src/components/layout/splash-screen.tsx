"use client";

import { Shield, Cpu, Car } from "lucide-react";
import { cn } from "@/lib/utils";
import Image from "next/image";

export function SplashScreen() {
  return (
    <div className="fixed inset-0 flex min-h-0 flex-col overflow-hidden bg-background">
      {/* Rings: cover full viewport so they stay centered */}
      <div
        className="pointer-events-none absolute inset-0 flex items-center justify-center overflow-hidden"
        aria-hidden
      >
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className={cn(
              "absolute rounded-full border border-primary/20",
              "animate-ping"
            )}
            style={{
              width: `${200 + i * 100}px`,
              height: `${200 + i * 100}px`,
              animationDelay: `${i * 0.5}s`,
              animationDuration: "3s",
            }}
          />
        ))}
      </div>

      {/* Main stack: scrolls on very short viewports; never overlaps footer */}
      <div className="relative z-10 flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-6 py-8">
        <div className="flex w-full max-w-md flex-col items-center gap-6 text-center">
          <div className="logo-animate shrink-0">
            <Image
              src="/logo.png"
              alt="ATHAR Logo"
              width={160}
              height={160}
              className="object-contain"
              priority
            />
          </div>

          <div className="shrink-0">
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              ATHAR
            </h1>
            <p className="mt-2 text-sm text-muted-foreground">
              City-Wide Vehicle Tracking System
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-3">
            <div className="flex gap-1">
              {[...Array(4)].map((_, i) => (
                <div
                  key={i}
                  className="h-2 w-2 rounded-full bg-primary animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
          </div>

          <div className="flex shrink-0 flex-wrap items-start justify-center gap-6 sm:gap-8">
            <FeatureBadge icon={Shield} label="Forensic" delay={0.3} />
            <FeatureBadge icon={Cpu} label="AI-Powered" delay={0.5} />
            <FeatureBadge icon={Car} label="Multi-Camera" delay={0.7} />
          </div>
        </div>
      </div>

      <footer className="relative z-10 shrink-0 border-t border-border/40 bg-background/80 px-4 py-4 text-center backdrop-blur-sm">
        <p className="text-xs leading-relaxed text-muted-foreground/70">
          Graduation Project 2024 — CityFlowV2 Demo
        </p>
      </footer>
    </div>
  );
}

function FeatureBadge({
  icon: Icon,
  label,
  delay,
}: {
  icon: React.ElementType;
  label: string;
  delay: number;
}) {
  return (
    <div
      className="flex flex-col items-center gap-1 animate-fade-in opacity-0"
      style={{
        animationDelay: `${delay}s`,
        animationFillMode: "forwards",
      }}
    >
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
        <Icon className="h-5 w-5 text-muted-foreground" />
      </div>
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}
