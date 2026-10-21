FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent ./agent
COPY bot ./bot
ENV PYTHONUNBUFFERED=1 LOG_DIR=/data/logs STATE_FILE=/data/logs/state.json KILL_FILE=/data/logs/KILL
CMD ["python", "-m", "bot", "--mode", "shadow"]
