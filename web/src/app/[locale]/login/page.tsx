"use client";

import { useLocale, useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { loginAuthLoginPost } from "@/lib/api";
import "@/lib/client";

export default function LoginPage() {
  const t = useTranslations();
  const locale = useLocale();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setFailed(false);
    const { response } = await loginAuthLoginPost({
      body: { username, password },
    });
    setBusy(false);
    if (response?.ok) {
      router.push(`/${locale}/runs`);
    } else {
      setFailed(true);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-sm space-y-4 rounded-xl border border-zinc-800 bg-zinc-900 p-8"
      >
        <div className="space-y-1 text-center">
          <h1 className="text-2xl font-bold">{t("app.title")}</h1>
          <p className="text-sm text-zinc-400">{t("app.subtitle")}</p>
        </div>
        <label className="block space-y-1">
          <span className="text-sm text-zinc-300">{t("login.username")}</span>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 outline-none focus:border-zinc-400"
          />
        </label>
        <label className="block space-y-1">
          <span className="text-sm text-zinc-300">{t("login.password")}</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 outline-none focus:border-zinc-400"
          />
        </label>
        {failed && (
          <p className="text-sm text-red-400" role="alert">
            {t("login.failed")}
          </p>
        )}
        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-md bg-zinc-100 px-3 py-2 font-semibold text-zinc-900 transition hover:bg-white disabled:opacity-50"
        >
          {t("login.submit")}
        </button>
      </form>
    </main>
  );
}
