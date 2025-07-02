#!/bin/bash
# start-services.sh - Script to start all SafeStreets services with correct ports

# Function to check if a port is in use
port_in_use() {
  lsof -i:$1 -t >/dev/null
  return $?
}

# Function to check if service is running by testing endpoint
check_service() {
  local name=$1
  local url=$2
  local max_tries=$3
  local try=1
  
  echo "Checking if $name is responding at $url..."
  while [ $try -le $max_tries ]; do
    if curl -s --head --request GET $url | grep "200 OK" > /dev/null; then
      echo "✓ $name is running and responding"
      return 0
    else
      echo "Waiting for $name to start (attempt $try/$max_tries)..."
      sleep 3
      try=$((try+1))
    fi
  done
  
  echo "✗ $name is not responding after $max_tries attempts"
  return 1
}

# Print header
echo "=========================================="
echo "SafeStreets Application Starter"
echo "=========================================="
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:5030"
echo "AI Models Server: http://localhost:5000"
echo "=========================================="

# Start backend server
if port_in_use 5030; then
  echo "Backend server is already running on port 5030"
else
  echo "Starting Backend Server on port 5030..."
  cd "$(dirname "$0")/backend"
  npm install &>/dev/null
  npm start &
  BACKEND_PID=$!
  echo "Backend started with PID: $BACKEND_PID"
  cd "$(dirname "$0")"
fi

# Check if backend is responding
check_service "Backend API" "http://localhost:5030/api/health" 10

# Start AI Models Server
if port_in_use 5000; then
  echo "AI Models server is already running on port 5000"
else
  echo "Starting AI Models Server on port 5000..."
  cd "$(dirname "$0")/ai_models_server"
  pip install -r requirements.txt &>/dev/null
  python app.py &
  AI_SERVER_PID=$!
  echo "AI Server started with PID: $AI_SERVER_PID"
  cd "$(dirname "$0")"
fi

# Start Admin Portal (Frontend)
if port_in_use 3000; then
  echo "Admin Portal is already running on port 3000"
else
  echo "Starting Admin Portal on port 3000..."
  cd "$(dirname "$0")/apps/admin-portal"
  npm install &>/dev/null
  PORT=3000 npm start &
  FRONTEND_PID=$!
  echo "Admin Portal started with PID: $FRONTEND_PID"
  cd "$(dirname "$0")"
fi

echo ""
echo "All services started successfully!"
echo "To start the mobile app, use VS Code task 'Start User App'"
echo "or run: cd apps/user-app && npx expo start"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to press Ctrl+C
trap "echo 'Stopping all services...'; kill $BACKEND_PID $AI_SERVER_PID $FRONTEND_PID; echo 'All services stopped.'; exit 0" INT
wait
