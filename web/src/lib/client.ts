import { client } from './api/client.gen';

// Same-box deployment by default; override with NEXT_PUBLIC_ATHAR_API.
// credentials: 'include' — auth is a server-side session cookie (httponly),
// there is no token for JS to hold. The default host MUST match how the
// app is opened (localhost vs 127.0.0.1 are different SITES — a SameSite
// cookie set by one is not sent on fetches from the other).
export const API_URL =
  process.env.NEXT_PUBLIC_ATHAR_API ?? 'http://localhost:8000';

client.setConfig({
  baseUrl: API_URL,
  credentials: 'include',
});

export { client };
