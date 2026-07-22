"use client";

import { useLocale, useTranslations } from "next-intl";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { logoutAuthLogoutPost, meAuthMeGet, type UserOut } from "@/lib/api";
import "@/lib/client";

// Client-side session guard: the cookie is httponly so the only way to know
// we are signed in is to ask the API. 401 -> login.
export default function AppShell({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const t = useTranslations();
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<UserOut | null>(null);

  useEffect(() => {
    let cancelled = false;
    meAuthMeGet().then(({ data, response }) => {
      if (cancelled) return;
      if (!response?.ok || !data) {
        router.replace(`/${locale}/login`);
      } else {
        setUser(data);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [locale, router, pathname]);

  async function logout() {
    await logoutAuthLogoutPost();
    router.replace(`/${locale}/login`);
  }

  if (!user) {
    return (
      <main className="flex min-h-screen items-center justify-center text-zinc-500">
        …
      </main>
    );
  }

  const tabs = [
    { href: `/${locale}/runs`, label: t("nav.runs") },
    { href: `/${locale}/jobs`, label: t("nav.jobs") },
  ];

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-zinc-800 px-6 py-3">
        <div className="flex items-center gap-6">
          <span className="text-lg font-bold">{t("app.title")}</span>
          <nav className="flex gap-4">
            {tabs.map((tab) => (
              <Link
                key={tab.href}
                href={tab.href}
                className={
                  pathname === tab.href
                    ? "font-semibold text-zinc-100"
                    : "text-zinc-400 transition hover:text-zinc-100"
                }
              >
                {tab.label}
              </Link>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-zinc-400">
            {user.username} · {user.role}
          </span>
          <button
            onClick={logout}
            className="rounded-md border border-zinc-700 px-3 py-1 text-zinc-300 transition hover:bg-zinc-800"
          >
            {t("nav.logout")}
          </button>
        </div>
      </header>
      <main className="p-6">{children}</main>
    </div>
  );
}
