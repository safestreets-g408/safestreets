# Deploying the AI Models Server to Render

This guide explains how to deploy the SafeStreets AI Models Server to Render.

## Prerequisites

1. A [Render](https://render.com) account
2. Access to your model files or their download URLs
3. Any required API keys (e.g., GEMINI_API_KEY)

## Deployment Options

There are two main ways to deploy to Render:

### Option 1: Using the Render Dashboard (Manual)

1. **Log in to Render** and go to your dashboard.

2. **Create a new Web Service**:
   - Click "New +" and select "Web Service"
   - Connect your GitHub/GitLab repository
   - Select the repository and branch

3. **Configure the service**:
   - Name: `safestreets-ai-models-server`
   - Environment: `Docker`
   - Build Command: (leave empty, Docker will handle this)
   - Start Command: (leave empty, specified in Dockerfile)

4. **Add environment variables**:
   - `FLASK_HOST`: `0.0.0.0`
   - `FLASK_PORT`: `10000` (Render will override this with $PORT)
   - `FLASK_DEBUG`: `false`
   - `GEMINI_API_KEY`: (your API key)
   - `LOG_LEVEL`: `INFO`
   - `CORS_ORIGINS`: `*` (or your specific origins)

5. **Add a persistent disk**:
   - Under "Disks", create a new disk
   - Mount path: `/app/models`
   - Size: 10 GB (adjust based on your model sizes)

6. **Deploy**:
   - Click "Create Web Service"

### Option 2: Using render.yaml (Blueprint)

1. **Push the repository** with the `render.yaml` file to GitHub/GitLab.

2. **Connect Render to your repository** if you haven't already.

3. **Create a Blueprint**:
   - Go to Blueprints in Render
   - Select "New Blueprint Instance"
   - Choose your repository
   - Configure any secrets needed (e.g., API keys)
   - Deploy

## Model Files

### Option 1: Store in Render Disk

After deploying:
1. SSH into your Render service
2. Upload model files to the `/app/models` directory

### Option 2: Download during Build

Update the `scripts/download_models_render.sh` script with the actual URLs to your model files, then they'll be downloaded during the Docker build process.

## Monitoring and Troubleshooting

- **View logs** in the Render dashboard
- **SSH into the service** for debugging:
  ```
  render ssh <service-name>
  ```
- **Check disk usage**:
  ```
  df -h
  ```

## Security Notes

- Never commit API keys to your repository
- Restrict CORS origins in production
- Consider using Render's environment groups for shared secrets

## Scaling

The AI Models Server is resource-intensive. Consider:
- Upgrading to a higher Render plan if needed
- Using a GPU instance if available
- Monitoring memory usage closely

For any issues, check the Render logs or contact Render support.
