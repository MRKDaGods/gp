import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export interface ActionsFooterProps {
  target?: ReactNode;
  run?: ReactNode;
  cancel?: ReactNode;
  continueAction?: ReactNode;
  children?: ReactNode;
  className?: string;
}

export function ActionsFooter({ target, run, cancel, continueAction, children, className }: ActionsFooterProps) {
  const hasContent = Boolean(target || run || cancel || continueAction || children);

  return (
    <footer
      className={cn(
        "sticky bottom-0 z-20 min-h-16 shrink-0 border-t bg-background/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80 sm:px-6",
        !hasContent && "hidden",
        className
      )}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">{target}</div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          {children}
          {cancel}
          {run}
          {continueAction}
        </div>
      </div>
    </footer>
  );
}
