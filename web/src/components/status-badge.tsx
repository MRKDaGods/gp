import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// One place for status -> color so runs, jobs, and hypotheses read the same.
const STATUS_CLASSES: Record<string, string> = {
  completed: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  confirmed: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  running: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
  claimed: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
  open: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
  queued: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  proposed: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  cancelled: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  failed: "bg-destructive/10 text-destructive dark:bg-destructive/20",
  rejected: "bg-destructive/10 text-destructive dark:bg-destructive/20",
};

export function StatusBadge({ status }: Readonly<{ status: string }>) {
  return (
    <Badge
      variant="secondary"
      className={cn("capitalize", STATUS_CLASSES[status])}
    >
      {status}
    </Badge>
  );
}
