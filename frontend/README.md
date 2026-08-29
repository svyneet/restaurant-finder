# Berlin Restaurant Kiezscout — frontend

SvelteKit chat UI for the [restaurant-finder](../README.md) multi-agent backend. Talks to the FastAPI backend's `/api/chat` SSE endpoint and renders a per-answer grounding report (citation pass/fail).

Scaffolded with [`sv`](https://github.com/sveltejs/cli):
```sh
npx sv@0.17.0 create --template minimal --types ts --install npm frontend
```

## Structure

- `src/lib/api.ts` — `streamChat()`: POSTs to `/api/chat`, streams "status" (pipeline stage progress) and "result" (final `RunResult` — answer, citations, tool calls) SSE events. Types mirror `src/agents/models.py` on the backend.
- `src/lib/chat.ts` — `ChatMessage` type + example prompts shown in the UI.
- `src/lib/GroundingReport.svelte` — renders citation grounding pass/fail for an answer.
- `src/routes/+page.svelte` — main chat page.

## Developing

Install dependencies once, then start the dev server:

```sh
npm install
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

The dev server proxies `/api` requests to the FastAPI backend — make sure it's running first (see the [root QUICKSTART.md](../QUICKSTART.md)):
```sh
.venv/bin/uvicorn backend.main:app --reload --port 8100
```

Then open http://localhost:5173.

## Building

To create a production version of your app:

```sh
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.

