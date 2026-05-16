"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ClipboardList, GitMerge, ScanSearch } from "lucide-react";
import { SplashScreen } from "@/components/layout/splash-screen";
import { MainDashboard } from "@/components/layout/main-dashboard";

export default function HomePage() {
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 2500);

    return () => clearTimeout(timer);
  }, []);

  if (showSplash) {
    return <SplashScreen />;
  }

  return (
    <div className="relative h-full min-h-0">
      <div className="absolute right-4 top-4 z-20 flex gap-2">
        <Link href="/reid" className="inline-flex h-9 items-center gap-2 rounded-md border bg-background/95 px-3 text-sm font-medium shadow-sm hover:bg-muted">
          <ScanSearch className="h-4 w-4" />
          ReID
        </Link>
        <Link href="/fusion" className="inline-flex h-9 items-center gap-2 rounded-md border bg-background/95 px-3 text-sm font-medium shadow-sm hover:bg-muted">
          <GitMerge className="h-4 w-4" />
          Fusion
        </Link>
        <Link href="/eval" className="inline-flex h-9 items-center gap-2 rounded-md border bg-background/95 px-3 text-sm font-medium shadow-sm hover:bg-muted">
          <ClipboardList className="h-4 w-4" />
          Eval
        </Link>
      </div>
      <MainDashboard />
    </div>
  );
}
