#!/bin/bash

mkdir -p /app/logs

# Start the Ollama server in the background and redirect output
ollama serve > /app/logs/ollama_serve.log 2>&1 &

# Function to check if the server is running
check_server() {
    curl -sSf http://ollama:11434 > /dev/null
}

llm_model=$(python3 -c '
import yaml
with open("python/euclid/rag/app_config.yaml", "r") as file:
    data = yaml.safe_load(file)
    print(data["llm"]["model"])
')
if [ -z "$llm_model" ]; then
    echo "Failed to parse config! Please check the config is valid."
    exit 1;
fi
echo "Container will use model: $llm_model"


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

# Run the Ollama command and redirect output
ollama pull $llm_model > /app/logs/ollama_pull.log 2>&1 &

# Start the Streamlit app
PYTHONPATH=/app/python streamlit run python/euclid/rag/app.py
