# RAGScope Frontend — Next.js 15
# Dev-oriented image (docker-compose mounts the source and runs `npm run dev`).
# The multi-stage production target builds a standalone server for Cloud Run.

# ── Base ──────────────────────────────────────
FROM node:22-alpine AS base
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

# ── Dependencies ──────────────────────────────
FROM base AS deps
COPY package.json package-lock.json* ./
RUN npm install

# ── Development (default) ─────────────────────
FROM base AS dev
COPY --from=deps /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]

# ── Production build ──────────────────────────
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM base AS prod
ENV NODE_ENV=production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
