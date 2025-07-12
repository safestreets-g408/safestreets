#!/bin/bash

# Deploy AI Models Server to Render
# Run this script to deploy the AI models server to Render

echo "🚀 Deploying AI Models Server to Render..."

# Check if Render CLI is installed
if ! command -v render &> /dev/null
then
    echo "❌ Render CLI is not installed. Please install it first:"
    echo "npm install -g @renderinc/cli"
    exit 1
fi

# Deploy using render.yaml configuration
echo "📦 Deploying using configuration in render.yaml..."
render deploy --yaml render.yaml

echo "✅ Deployment initiated! Check your Render dashboard for progress."
