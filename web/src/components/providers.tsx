"use client";

import { DirectionProvider } from "@base-ui/react/direction-provider";
import { ThemeProvider } from "next-themes";
import { TooltipProvider } from "@/components/ui/tooltip";

// DirectionProvider makes Base UI popups/menus RTL-aware (the html dir
// attribute alone only affects layout, not their positioning logic).
export function Providers({
  dir,
  children,
}: Readonly<{ dir: "ltr" | "rtl"; children: React.ReactNode }>) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <DirectionProvider direction={dir}>
        <TooltipProvider>{children}</TooltipProvider>
      </DirectionProvider>
    </ThemeProvider>
  );
}
