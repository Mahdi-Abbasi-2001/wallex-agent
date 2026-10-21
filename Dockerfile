FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent ./agent
COPY bot ./bot
COPY scripts ./scripts
ARG GIT_COMMIT=unknown
# Logs, state and open positions live on the persistent disk mounted at /data (see liara.json).
ENV PYTHONUNBUFFERED=1 GIT_COMMIT=$GIT_COMMIT LOG_DIR=/data/logs STATE_FILE=/data/logs/state.json KILL_FILE=/data/logs/KILL PORT=8000
EXPOSE 8000
# BOT_MODE=shadow (default) or live; live also needs BOT_FLAGS=--confirm-live and CONFIRM_LIVE=yes in the app's environment.
CMD ["sh", "-c", "mkdir -p /data/logs && exec python -m bot --mode ${BOT_MODE:-shadow} ${BOT_FLAGS:-}"]
