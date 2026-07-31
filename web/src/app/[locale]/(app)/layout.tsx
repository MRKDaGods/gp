"use client";

import { useLocale, useTranslations } from "next-intl";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ThemeToggle } from "@/components/theme-toggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { logoutAuthLogoutPost, meAuthMeGet, type UserOut } from "@/lib/api";
import "@/lib/client";
import { cn } from "@/lib/utils";

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
      <main className="mx-auto max-w-6xl space-y-4 p-6">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-32 w-full" />
      </main>
    );
  }

  const tabs = [
    { href: `/${locale}/ingest`, label: t("nav.ingest") },
    { href: `/${locale}/cases`, label: t("nav.cases") },
    { href: `/${locale}/runs`, label: t("nav.runs") },
    { href: `/${locale}/jobs`, label: t("nav.jobs") },
  ];

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-6">
            <span className="text-lg font-bold">{t("app.title")}</span>
            <nav className="flex gap-1">
              {tabs.map((tab) => (
                <Link
                  key={tab.href}
                  href={tab.href}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-sm transition-colors",
                    pathname.startsWith(tab.href)
                      ? "bg-muted font-medium text-foreground"
                      : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                  )}
                >
                  {tab.label}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-muted-foreground">{user.username}</span>
            <Badge variant="outline">{user.role}</Badge>
            <ThemeToggle />
            <Button variant="outline" size="sm" onClick={logout}>
              {t("nav.logout")}
            </Button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl p-6">{children}</main>
    </div>
  );
}
