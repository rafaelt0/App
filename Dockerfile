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

# FinBERT (torch + transformers) excede os 512 MB do plano gratuito do Render.
# Sem ele, as notícias usam o fallback léxico identificado na interface.
ARG WITH_FINBERT=0
COPY requirements.txt ./
RUN if [ "$WITH_FINBERT" = "1" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt; \
    else \
        grep -Ev '^(torch|transformers)==' requirements.txt > requirements-deploy.txt && \
        pip install -r requirements-deploy.txt; \
    fi

COPY . .
COPY --from=frontend /app/frontend/.next ./frontend/.next
COPY --from=frontend /app/frontend/node_modules ./frontend/node_modules

ENV B3LAB_ENV=production \
    B3LAB_HOST=0.0.0.0
EXPOSE 10000
# O Render define PORT e RENDER_EXTERNAL_URL; B3LAB_ORIGIN explícito tem prioridade.
CMD ["sh", "-c", "B3LAB_PORT=${PORT:-10000} B3LAB_ORIGIN=${B3LAB_ORIGIN:-$RENDER_EXTERNAL_URL} exec python run.py --production"]
