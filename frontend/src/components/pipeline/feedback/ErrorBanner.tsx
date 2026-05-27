import { AlertCircle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";

import { StageStatusBadge } from "../status/StageStatusBadge";

export interface ErrorBannerProps {
  title?: string;
  message?: string | null;
  actions?: React.ReactNode;
  className?: string;
}

export function ErrorBanner({ title = "Something went wrong", message, actions, className }: ErrorBannerProps) {
  if (!message) return null;

  return (
    <Alert variant="destructive" className={cn("items-start", className)}>
      <AlertCircle className="h-4 w-4" />
      <AlertTitle className="flex flex-wrap items-center gap-2">
        <StageStatusBadge status="error" label="Error" />
        <span>{title}</span>
      </AlertTitle>
      <AlertDescription className="mt-2 space-y-3">
        <div>{message}</div>
        {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
      </AlertDescription>
    </Alert>
  );
}
