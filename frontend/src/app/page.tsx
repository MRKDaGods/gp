"use client";

import { useState, useEffect } from "react";
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
    return (
      <div className="relative h-full min-h-0">
        <SplashScreen />
      </div>
    );
  }

  return (
    <div className="relative h-full min-h-0">
      <MainDashboard />
    </div>
  );
}
