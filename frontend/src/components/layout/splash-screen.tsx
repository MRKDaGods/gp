"use client";

import { Shield, Cpu, Car } from "lucide-react";
import { cn } from "@/lib/utils";
import Image from "next/image";

export function SplashScreen() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-background">
      <div className="relative flex flex-col items-center">
        {/* Animated background rings */}
        <div className="absolute inset-0 flex items-center justify-center">
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

        {/* Logo - Using actual PNG */}
        <div className="relative z-10 logo-animate">
          <Image
            src="/logo.png"
            alt="ATHAR Logo"
            width={160}
            height={160}
            className="object-contain"
            priority
          />
        </div>

        {/* Title */}
        <div className="relative z-10 mt-6 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">
            ATHAR
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            City-Wide Vehicle Tracking System
          </p>
        </div>

        {/* Loading indicator */}
        <div className="relative z-10 mt-8 flex items-center gap-3">
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

        {/* Feature badges */}
        <div className="relative z-10 mt-12 flex gap-6">
          <FeatureBadge icon={Shield} label="Forensic" delay={0.3} />
          <FeatureBadge icon={Cpu} label="AI-Powered" delay={0.5} />
          <FeatureBadge icon={Car} label="Multi-Camera" delay={0.7} />
        </div>

        {/* Version */}
        <p className="absolute bottom-8 text-xs text-muted-foreground/50">
          Graduation Project 2024 - CityFlowV2 Demo
        </p>
      </div>
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
