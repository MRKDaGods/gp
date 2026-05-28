import { AlertCircle, AlertTriangle, Info } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";

import { StageStatusBadge } from "../status/StageStatusBadge";

export interface ErrorBannerProps {
  title?: string;
  message?: string | null;
  severity?: "error" | "warning" | "info";
  actions?: React.ReactNode;
  className?: string;
}

const SEVERITY_STYLES = {
  error: {
    icon: AlertCircle,
    badgeStatus: "error" as const,
    badgeLabel: "Error",
    alertClassName: "border-destructive/40 bg-destructive/10 text-destructive [&>svg]:text-destructive",
  },
  warning: {
    icon: AlertTriangle,
    badgeStatus: "stale" as const,
    badgeLabel: "Warning",
    alertClassName: "border-warning/40 bg-warning/10 text-warning [&>svg]:text-warning",
  },
  info: {
    icon: Info,
    badgeStatus: "idle" as const,
    badgeLabel: "Info",
    alertClassName: "border-info/40 bg-info/10 text-info [&>svg]:text-info",
  },
};

export function ErrorBanner({ title = "Something went wrong", message, severity = "error", actions, className }: ErrorBannerProps) {
  if (!message) return null;
  const style = SEVERITY_STYLES[severity];
  const Icon = style.icon;

  return (
    <Alert variant="default" className={cn("items-start", style.alertClassName, className)}>
      <Icon className="h-4 w-4" />
      <AlertTitle className="flex flex-wrap items-center gap-2">
        <StageStatusBadge status={style.badgeStatus} label={style.badgeLabel} />
        <span>{title}</span>
      </AlertTitle>
      <AlertDescription className="mt-2 space-y-3">
        <div className="whitespace-pre-line">{message}</div>
        {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
      </AlertDescription>
    </Alert>
  );
}
