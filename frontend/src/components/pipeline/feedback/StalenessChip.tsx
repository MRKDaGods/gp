import { StageStatusBadge } from "../status/StageStatusBadge";

export interface StalenessChipProps {
  label?: string;
}

export function StalenessChip({ label = "Stale - needs re-run" }: StalenessChipProps) {
  return <StageStatusBadge status="stale" label={label} />;
}
