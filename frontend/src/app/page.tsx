"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ClipboardList, GitMerge, ScanSearch } from "lucide-react";
import { SplashScreen } from "@/components/layout/splash-screen";
import { MainDashboard } from "@/components/layout/main-dashboard";

function Phase2Navigation() {
  return (
    <nav aria-label="Phase 2" className="absolute right-4 top-4 z-20 flex gap-2">
      <Link href="/reid" className="inline-flex h-9 items-center gap-2 rounded-md border border-border/60 bg-background/95 px-3 text-sm font-medium shadow-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
        <ScanSearch className="h-4 w-4" />
        ReID
      </Link>
      <Link href="/fusion" className="inline-flex h-9 items-center gap-2 rounded-md border border-border/60 bg-background/95 px-3 text-sm font-medium shadow-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
        <GitMerge className="h-4 w-4" />
        Fusion
      </Link>
      <Link href="/eval" className="inline-flex h-9 items-center gap-2 rounded-md border border-border/60 bg-background/95 px-3 text-sm font-medium shadow-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
        <ClipboardList className="h-4 w-4" />
        Eval
      </Link>
    </nav>
  );
}

export default function HomePage() {
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 2500);

    return () => clearTimeout(timer);
  }, []);

  if (showSplash) {
    return (
      <div className="relative h-full min-h-0">
        <SplashScreen />
        <Phase2Navigation />
      </div>
    );
  }

  return (
    <div className="relative h-full min-h-0">
      <Phase2Navigation />
      <MainDashboard />
    </div>
  );
}
