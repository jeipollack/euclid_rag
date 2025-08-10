#!/bin/bash

mkdir -p /app/logs

# Start the Ollama server in the background and redirect output
ollama serve > /app/logs/ollama_serve.log 2>&1 &

# Function to check if the server is running
check_server() {
    curl -sSf http://localhost:11434 > /dev/null
}

# Wait until the server is running
timeout=30
count=0
until check_server; do
    if [ $count -ge $timeout ]; then
        echo "Ollama server failed to start after $timeout seconds."
        kill $(cat "$PID_FILE") || true
        exit 1
    fi
    echo "Waiting for Ollama server to start..."
    sleep 1
    count=$((count + 1))
done

ollama pull gemma3:4b > /app/logs/ollama_pull.log 2>&1 &

echo "Running Ollama server."
while :
do
	sleep 100
done