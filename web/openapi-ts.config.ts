import { defineConfig } from '@hey-api/openapi-ts';

// Regenerate after API changes:
//   (repo root) .venv-v2/Scripts/python -c "..." > web/openapi.json  — see README
//   (web/)      pnpm generate:api
export default defineConfig({
  input: 'openapi.json',
  output: 'src/lib/api',
  plugins: ['@hey-api/client-fetch'],
});
