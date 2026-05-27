"use client";

import { ExecutionTargetToggle } from "@/components/pipeline/run/ExecutionTargetToggle";

export interface KaggleExecutionToggleProps {
  stage: number;
  className?: string;
}

export function KaggleExecutionToggle({ stage, className }: KaggleExecutionToggleProps) {
  return <ExecutionTargetToggle stage={stage} className={className} />;
}