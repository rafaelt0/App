# Next.js e API Python no mesmo contêiner, iniciados por run.py.
FROM node:22-bookworm-slim AS frontend
WORKDIR /app/frontend
ENV NEXT_TELEMETRY_DISABLED=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=frontend /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
COPY --from=frontend /app/frontend/.next ./frontend/.next
COPY --from=frontend /app/frontend/node_modules ./frontend/node_modules

ENV B3LAB_ENV=production \
    B3LAB_HOST=0.0.0.0
EXPOSE 10000
# O Render define PORT e RENDER_EXTERNAL_URL; B3LAB_ORIGIN e B3LAB_KEEPALIVE_URL explícitos têm prioridade.
CMD ["sh", "-c", "B3LAB_PORT=${PORT:-10000} B3LAB_ORIGIN=${B3LAB_ORIGIN:-$RENDER_EXTERNAL_URL} B3LAB_KEEPALIVE_URL=${B3LAB_KEEPALIVE_URL-$RENDER_EXTERNAL_URL} exec python run.py --production"]
