# MTMC Tracker Frontend

Modern Next.js dashboard for the City-Wide Camera Vehicle Tracking System.

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Features

- Dark theme UniFi-style security dashboard
- 6-stage pipeline workflow
- Real-time progress tracking via WebSocket
- Interactive tracklet timeline (Clipchamp-style)
- Multi-camera split-screen video player
- Egypt location hierarchy filters
- Drag-and-drop video upload
- Responsive design (mobile-friendly)

## Tech Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Zustand (state management)
- TanStack Query (data fetching)
- Video.js (video playback)
- Leaflet (maps - future)

## Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Project Structure

```
src/
├── app/                 # Next.js pages
├── components/
│   ├── layout/          # Dashboard layout
│   ├── stages/          # Pipeline stage views
│   └── ui/              # shadcn components
├── lib/                 # API client, utilities
├── store/               # Zustand stores
├── types/               # TypeScript types
└── hooks/               # Custom React hooks
```

## Development

```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Build
npm run build
```

## License

Graduation Project - Educational Use Only
