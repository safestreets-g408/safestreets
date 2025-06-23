#!/bin/bash
# start-services.sh - Script to start all SafeStreets services with correct ports

# Print header
echo "=========================================="
echo "SafeStreets Application Starter"
echo "=========================================="
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:5030"
echo "AI Models Server: http://localhost:5000"
echo "=========================================="

# Start backend server
echo "Starting Backend Server on port 5030..."
cd "$(dirname "$0")/backend"
npm install &>/dev/null
npm start &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Start AI Models Server
echo "Starting AI Models Server on port 5000..."
cd "$(dirname "$0")/ai_models_server"
pip install -r requirements.txt &>/dev/null
python app.py &
AI_SERVER_PID=$!
echo "AI Server started with PID: $AI_SERVER_PID"

# Start Admin Portal (Frontend)
echo "Starting Admin Portal on port 3000..."
cd "$(dirname "$0")/apps/admin-portal"
npm install &>/dev/null
PORT=3000 npm start &
FRONTEND_PID=$!
echo "Admin Portal started with PID: $FRONTEND_PID"

echo "All services started successfully!"
echo "Press Ctrl+C to stop all services"

# Wait for user to press Ctrl+C
trap "echo 'Stopping all services...'; kill $BACKEND_PID $AI_SERVER_PID $FRONTEND_PID; echo 'All services stopped.'; exit 0" INT
wait
