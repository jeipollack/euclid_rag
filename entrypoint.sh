#!/bin/bash

mkdir -p /app/logs

# Start the Ollama server in the background and redirect output
ollama serve > /app/logs/ollama_serve.log 2>&1 &

# Function to check if the server is running
check_server() {
    curl -sSf http://localhost:11434 > /dev/null
}

# Wait until the server is running
until check_server; do
    echo "Waiting for Ollama server to start..."
    sleep 2
done

# Run the Ollama command and redirect output
ollama run gemma3:4b > /app/logs/ollama_run.log 2>&1 &

# Start the Streamlit app
PYTHONPATH=/app/python streamlit run python/euclid/rag/app.py
