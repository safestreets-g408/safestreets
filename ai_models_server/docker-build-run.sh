#!/bin/bash

# Script to build and run the AI Models Server Docker container
# Usage: ./docker-build-run.sh [--build] [--detach]

# Default options
BUILD=false
DETACHED=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --build)
      BUILD=true
      shift
      ;;
    --detach|-d)
      DETACHED=true
      shift
      ;;
    *)
      # Unknown option
      echo "Unknown option: $arg"
      echo "Usage: ./docker-build-run.sh [--build] [--detach]"
      exit 1
      ;;
  esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️ Docker Compose is not installed. Falling back to Docker CLI."
    USE_COMPOSE=false
else
    USE_COMPOSE=true
fi

# Create directories if they don't exist
mkdir -p static/uploads static/results

if [ "$BUILD" = true ]; then
    echo "🔨 Building Docker image..."
    if [ "$USE_COMPOSE" = true ]; then
        docker-compose build
    else
        docker build -t safestreets-ai-server .
    fi
fi

echo "🚀 Starting AI Models Server..."
if [ "$USE_COMPOSE" = true ]; then
    if [ "$DETACHED" = true ]; then
        docker-compose up -d
        echo "✅ AI Models Server started in detached mode. Access at http://localhost:5000"
        echo "📋 To view logs, run: docker-compose logs -f"
    else
        docker-compose up
    fi
else
    if [ "$DETACHED" = true ]; then
        docker run -d -p 5000:5000 \
            -v "$(pwd)/static/uploads:/app/static/uploads" \
            -v "$(pwd)/static/results:/app/static/results" \
            --name safestreets-ai-server \
            safestreets-ai-server
        echo "✅ AI Models Server started in detached mode. Access at http://localhost:5000"
        echo "📋 To view logs, run: docker logs -f safestreets-ai-server"
    else
        docker run -p 5000:5000 \
            -v "$(pwd)/static/uploads:/app/static/uploads" \
            -v "$(pwd)/static/results:/app/static/results" \
            --name safestreets-ai-server \
            safestreets-ai-server
    fi
fi
