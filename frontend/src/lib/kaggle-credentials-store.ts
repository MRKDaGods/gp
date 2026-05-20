"use client";

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface KaggleCredentials {
  username: string;
  key: string;
}

export interface KaggleCredentialsState {
  credentials: KaggleCredentials | null;
  setCredentials: (creds: KaggleCredentials | null) => void;
  clearCredentials: () => void;
}

/**
 * Browser-only Kaggle credentials cache.
 *
 * The project intentionally allows localStorage persistence for user-supplied Kaggle tokens:
 * they stay in this browser, will be sent to the backend per request in Phase 13, and are
 * never written to backend disk. Keep this aligned with the Settings modal warning text.
 */
export const useKaggleCredentialsStore = create<KaggleCredentialsState>()(
  persist(
    (set) => ({
      credentials: null,
      setCredentials: (credentials) => set({ credentials }),
      clearCredentials: () => set({ credentials: null }),
    }),
    {
      name: 'mtmc-kaggle-credentials',
      version: 1,
    },
  ),
);

export const useHasKaggleCredentials = () =>
  useKaggleCredentialsStore((s) => s.credentials !== null);
