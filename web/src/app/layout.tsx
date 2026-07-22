// Root layout is a passthrough; <html dir/lang> lives in [locale]/layout.tsx
// so RTL/LTR is decided per request (next-intl app-router pattern).
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
