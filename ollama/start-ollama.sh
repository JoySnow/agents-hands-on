#!/bin/bash

# Ollama Startup Script
# This script loads the environment configuration and starts Ollama

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables
source "$SCRIPT_DIR/ollama.env"

echo "Starting Ollama with OLLAMA_HOST=$OLLAMA_HOST"
echo "Accessible at: http://0.0.0.0:11434"
echo "From network: http://$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}'):11434"
echo ""

# Start Ollama
ollama serve
