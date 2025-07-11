FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p static/uploads static/results models

# Download model files if needed (make script executable and run it)
RUN chmod +x scripts/download_models_render.sh && ./scripts/download_models_render.sh

# Set up environment variables
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=10000
ENV FLASK_DEBUG=False

# Expose the port the app will run on
EXPOSE 10000

# Command to run the application with gunicorn
# We use the $PORT environment variable provided by Render
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} "app:main()" --timeout 120 --workers 1
