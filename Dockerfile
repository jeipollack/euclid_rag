# ================================ Python build stage ================================

FROM python:3.12-slim-bookworm AS builder

# Get latest uv (TODO: use tagged version)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git

# Set the working directory in the container
WORKDIR /app

# Install additional package requirements, attempt a dummy install and then a full install
COPY README.md LICENSE pyproject.toml /app/
COPY python/euclid /app/python/euclid
RUN SETUPTOOLS_SCM_PRETEND_VERSION="0.0.1" uv sync
RUN --mount=source=.git,target=.git,type=bind uv pip install -e .

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# # ================================ Prod stage ================================

FROM python:3.12-slim-bookworm AS production

WORKDIR /app

# Copy necessary source to container
COPY python/euclid /app/python/euclid
COPY --from=builder /app/.venv .venv
COPY --from=builder /app/python/euclid/_version.py /app/python/euclid/_version.py

ENV PATH="/app/.venv/bin:$PATH"

# Expose the port that the app will run on
EXPOSE 8501

# Run the application when the container launches
ENV PYTHONPATH=/app/python
CMD ["streamlit", "run", "python/euclid/rag/app.py"]
