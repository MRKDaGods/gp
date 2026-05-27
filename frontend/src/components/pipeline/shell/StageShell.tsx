"use client";

import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { ActionsFooter, type ActionsFooterProps } from "./ActionsFooter";
import { ContractBanner, type ContractBannerProps } from "./ContractBanner";

export interface StageShellProps {
  contract: ContractBannerProps;
  actions?: ActionsFooterProps;
  children?: ReactNode;
  className?: string;
  contentClassName?: string;
}

export function StageShell({ contract, actions, children, className, contentClassName }: StageShellProps) {
  return (
    <section className={cn("flex h-full min-h-0 flex-col overflow-hidden", className)}>
      <ContractBanner {...contract} />
      <div className={cn("flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden", contentClassName)}>{children}</div>
      <ActionsFooter {...actions} />
    </section>
  );
}
