#!/bin/bash

# Script to build and push the AI Models Server Docker image to Docker Hub
# Usage: ./docker-push.sh [--build] [--username your-username] [--tag tagname]

# Default options
BUILD=true
USERNAME=""
TAG="latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      BUILD=true
      shift
      ;;
    --no-build)
      BUILD=false
      shift
      ;;
    --username)
      USERNAME="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    *)
      # Unknown option
      echo "Unknown option: $1"
      echo "Usage: ./docker-push.sh [--build] [--username your-username] [--tag tagname]"
      exit 1
      ;;
  esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# If username wasn't provided, ask for it
if [ -z "$USERNAME" ]; then
    echo "🔑 Please enter your Docker Hub username:"
    read -r USERNAME
    if [ -z "$USERNAME" ]; then
        echo "❌ Username cannot be empty."
        exit 1
    fi
fi

# Define the image name
IMAGE_NAME="${USERNAME}/safestreets-ai-server:${TAG}"

if [ "$BUILD" = true ]; then
    echo "🔨 Building Docker image as ${IMAGE_NAME}..."
    docker build -t "$IMAGE_NAME" .
    
    if [ $? -ne 0 ]; then
        echo "❌ Docker build failed."
        exit 1
    fi
    echo "✅ Docker image built successfully."
fi

echo "🔐 Logging in to Docker Hub..."
echo "Please enter your Docker Hub credentials when prompted:"
docker login

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed."
    exit 1
fi

echo "🚀 Pushing image to Docker Hub as ${IMAGE_NAME}..."
docker push "$IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed."
    exit 1
fi

echo "✅ Successfully pushed ${IMAGE_NAME} to Docker Hub."
echo "🌐 Your image is now available at: https://hub.docker.com/r/${USERNAME}/safestreets-ai-server"
