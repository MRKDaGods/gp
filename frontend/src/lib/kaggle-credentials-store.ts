"use client";

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface KaggleCredentials {
  username: string;
  key: string;
}

export interface KaggleCredentialsState {
  credentials: KaggleCredentials | null;
  modalOpen: boolean;
  setCredentials: (creds: KaggleCredentials | null) => void;
  clearCredentials: () => void;
  setModalOpen: (open: boolean) => void;
  openCredentialsModal: () => void;
}

/** Browser-only Kaggle credentials cache. */
export const useKaggleCredentialsStore = create<KaggleCredentialsState>()(
  persist(
    (set) => ({
      credentials: null,
      modalOpen: false,
      setCredentials: (credentials) => set({ credentials }),
      clearCredentials: () => set({ credentials: null }),
      setModalOpen: (modalOpen) => set({ modalOpen }),
      openCredentialsModal: () => set({ modalOpen: true }),
    }),
    {
      name: 'mtmc-kaggle-credentials',
      version: 1,
      // Persist only the credentials. `modalOpen` is transient UI state - persisting
      // it would reopen the modal on every reload if it happened to be open.
      partialize: (s) => ({ credentials: s.credentials }),
    },
  ),
);

export const useHasKaggleCredentials = () =>
  useKaggleCredentialsStore((s) => s.credentials !== null);
