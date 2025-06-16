# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy pyproject.toml separately to leverage Docker cache
COPY pyproject.toml /app/

# Install required packages
RUN uv pip install --system -r pyproject.toml

# Copy necessary source to container
COPY . /app

# Entrypoint needs to executable
RUN chmod +x entrypoint.sh

# Expose the port that the app will run on
EXPOSE 8501

# Run the application when the container launches
ENTRYPOINT ["./entrypoint.sh"]
