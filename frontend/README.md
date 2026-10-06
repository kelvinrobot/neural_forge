# Neural Forge – AI Visual Diagnostics (frontend)

A static single-page app (Vite, React 18, TypeScript, Tailwind CSS, Framer Motion, MediaPipe Tasks Vision) for the Neural Forge camera-based examination system. It has no server of its own: it talks to the deployed backend over HTTPS and a WebSocket.

## Requirements

- Node.js 20 or newer and npm.
- A browser with camera access (Chrome, Edge, Safari or Firefox, desktop or mobile).

## Environment variable

| Variable | Default | Description |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `https://drrobot9-neural-forge.hf.space` | Base URL of the backend, without a trailing slash. The WebSocket URL is derived from it (`https` becomes `wss`). |

Copy `.env.example` to `.env` to override it locally. On Netlify or Vercel, set it in the project's environment variables. Vite reads it at build time, so redeploy after changing it.

## Local build

```bash
cd frontend
npm install
npm run build      # type-checks and writes the production build to dist/
npm run preview    # serves dist/ locally
```

## HTTPS is required for the camera

Browsers only allow camera access on secure origins (`https://` or `http://localhost`). Netlify and Vercel serve every deployment over HTTPS. If you open the app any other way, the camera step shows a clear message instead of a preview.

## Deploy to Netlify

1. In Netlify, choose **Add new site → Import an existing project** and select the repository.
2. Set **Base directory** to `frontend`. The build command (`npm run build`) and publish directory (`dist`) come from `frontend/netlify.toml`.
3. Under **Site configuration → Environment variables**, add `VITE_API_BASE_URL` if you use a different backend.
4. Deploy. `netlify.toml` also rewrites every route to `index.html`, so links such as `/report/<id>` work on refresh.

## Deploy to Vercel

1. In Vercel, choose **Add New → Project** and import the repository.
2. Set **Root Directory** to `frontend`. Vercel detects Vite. `frontend/vercel.json` sets the build command, output directory (`dist`) and the single-page-app rewrite.
3. Add `VITE_API_BASE_URL` under **Settings → Environment Variables** if needed.
4. Deploy.

## Notes

- In-browser landmark overlays use MediaPipe models loaded from the official MediaPipe CDN (`cdn.jsdelivr.net` and `storage.googleapis.com`). They are only drawn on screen; the backend receives plain camera frames.
- Frames are sent unmirrored at about 20–25 frames per second. They are dropped when the connection is congested.
- Patient details and reports are kept in memory only. Nothing is written to `localStorage` or cookies. Reports can be printed, saved as PDF or downloaded as JSON from the report page.
- If the backend is waking from sleep, the home page shows a status screen and checks again every 5 seconds.
