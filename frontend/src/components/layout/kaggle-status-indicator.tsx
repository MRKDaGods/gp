"use client";

import { useHasKaggleCredentials } from '@/lib/kaggle-credentials-store';

export function KaggleCredentialsStatus() {
  const hasCredentials = useHasKaggleCredentials();

  if (hasCredentials) {
    return <span className="text-green-600">● Configured</span>;
  }

  return (
    <span className="text-muted-foreground italic">
      Not configured (using server default)
    </span>
  );
}
