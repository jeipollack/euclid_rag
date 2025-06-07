# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN apt-get -y install git curl
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy necessary source to container
COPY . /app

# Entrypoint needs to executable
RUN chmod +x entrypoint.sh

# Install any needed packages specified in requirements.txt
RUN uv pip install --system --upgrade pip
RUN uv pip install --system .

# Expose the port that the app will run on
EXPOSE 8501

# Run the application when the container launches
ENTRYPOINT ["./entrypoint.sh"]
