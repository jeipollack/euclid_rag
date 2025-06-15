# ================================ Ollama build stage ================================

FROM python:3.12-slim-bookworm AS ollama


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# ================================ Python build stage ================================

FROM python:3.12-slim-bookworm AS builder

# Get latest uv (TODO: use tagged version)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends

# Set the working directory in the container
WORKDIR /app

# Copy pyproject.toml separately to leverage Docker cache
COPY ./pyproject.toml /app/
COPY python/euclid /app/python/euclid

RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EUCLID_RAG="0.0.1" uv sync
# RUN --mount=source=.git,target=.git,type=bind uv sync

# Install required packages
RUN uv sync

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# # ================================ Prod stage ================================

FROM ollama AS production

WORKDIR /app

COPY . .

# Copy necessary source to container
COPY python/euclid entrypoint.sh /app/
COPY --from=builder /app/.venv .venv

ENV PATH="/app/.venv/bin:$PATH"

# Entrypoint needs to executable
RUN chmod +x entrypoint.sh

# Expose the port that the app will run on
EXPOSE 8501

# Run the application when the container launches
ENTRYPOINT ["./entrypoint.sh"]
