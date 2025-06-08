#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Function to prompt for and set environment variable
set_env_var() {
    local var_name=$1
    local prompt=$2
    local current_value=$(grep "^$var_name=" .env | cut -d'=' -f2)
    
    if [ -n "$current_value" ]; then
        read -p "$prompt [$current_value]: " value
        value=${value:-$current_value}
    else
        read -p "$prompt: " value
    fi
    
    # Update or add the variable in .env
    if grep -q "^$var_name=" .env; then
        sed -i '' "s|^$var_name=.*|$var_name=$value|" .env
    else
        echo "$var_name=$value" >> .env
    fi
    
    # Export for current session
    export $var_name=$value
}

# Set up environment variables
set_env_var "OPENAI_API_KEY" "Enter your OpenAI API key"
set_env_var "QDRANT_URL" "Enter your Qdrant URL"
set_env_var "QDRANT_API_KEY" "Enter your Qdrant API key"

echo "Environment variables have been set up!"
echo "You can now run the chat application with: source venv/bin/activate && python chat.py" 