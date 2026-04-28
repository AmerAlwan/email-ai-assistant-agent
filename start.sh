#!/bin/bash
set -e

# Start the FastAPI HTTP server in the background
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!

# Start the LiveKit agent worker in dev mode (connects to LiveKit server and waits for jobs)
uv run python agent.py dev &
AGENT_PID=$!

# If either process exits, kill the other and exit
wait -n
kill $UVICORN_PID $AGENT_PID 2>/dev/null
