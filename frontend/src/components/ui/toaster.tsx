"use client";

import * as React from "react";
import { AlertCircle, CheckCircle2, Info } from "lucide-react";
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast";
import { useToast } from "@/hooks/use-toast";

const VARIANT_ICON = {
  success: <CheckCircle2 className="h-5 w-5 text-success" />,
  destructive: <AlertCircle className="h-5 w-5 text-destructive" />,
  default: <Info className="h-5 w-5 text-accent-strong" />,
} as const;

export function Toaster() {
  const { toasts } = useToast();

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, variant, ...props }) {
        const icon = VARIANT_ICON[(variant as keyof typeof VARIANT_ICON) ?? "default"] ?? VARIANT_ICON.default;
        return (
          <Toast key={id} variant={variant} {...props}>
            <span className="mt-0.5 shrink-0">{icon}</span>
            <div className="grid min-w-0 flex-1 gap-0.5">
              {title && <ToastTitle>{title}</ToastTitle>}
              {description && (
                <ToastDescription>{description}</ToastDescription>
              )}
            </div>
            {action}
            <ToastClose />
          </Toast>
        );
      })}
      <ToastViewport />
    </ToastProvider>
  );
}
