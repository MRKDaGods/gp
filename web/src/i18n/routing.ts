import { defineRouting } from 'next-intl/routing';

// Arabic is the DEFAULT locale (primary deployment audience); English is the
// engineering/dev locale. RTL is derived from the locale in the layout.
export const routing = defineRouting({
  locales: ['ar', 'en'],
  defaultLocale: 'ar',
});

export type AppLocale = (typeof routing.locales)[number];

export function dirFor(locale: string): 'rtl' | 'ltr' {
  return locale === 'ar' ? 'rtl' : 'ltr';
}
