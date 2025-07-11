# Docker Support for AI Models Server

This AI Models Server now supports Docker for easy deployment and containerization.

## 🐳 Docker Setup

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier management)
- Docker Hub account (for pushing images)

### Using Docker

1. **Build the Docker image**:
   ```bash
   cd ai_models_server
   docker build -t safestreets-ai-server .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 5000:5000 safestreets-ai-server
   ```

### Using Docker Compose

1. **Start the service**:
   ```bash
   cd ai_models_server
   docker-compose up
   ```

2. **Start in detached mode** (runs in background):
   ```bash
   docker-compose up -d
   ```

3. **View logs**:
   ```bash
   docker-compose logs -f
   ```

4. **Stop the service**:
   ```bash
   docker-compose down
   ```

## Environment Variables

You can configure the service by setting environment variables in the docker-compose.yml file:

```yaml
environment:
  - FLASK_HOST=0.0.0.0
  - FLASK_PORT=5000
  - FLASK_DEBUG=False
  - GEMINI_API_KEY=your_api_key
```

## Volumes

The Docker setup includes volumes for persistent storage:
- `./static/uploads:/app/static/uploads` - For uploaded images
- `./static/results:/app/static/results` - For analysis results

## Pushing to Docker Hub

You can push your Docker image to Docker Hub to make it available for deployment anywhere:

1. **Build and push in one step**:
   ```bash
   ./docker-push.sh --username your-username
   ```

2. **Push with a specific tag**:
   ```bash
   ./docker-push.sh --username your-username --tag v1.0.0
   ```

3. **Push without rebuilding**:
   ```bash
   ./docker-push.sh --no-build --username your-username
   ```

4. **Pull the image on another machine**:
   ```bash
   docker pull your-username/safestreets-ai-server:latest
   ```

## Notes

- Model files are included in the Docker image
- Make sure to provide API keys if needed
- For production use, consider using a reverse proxy like Nginx
