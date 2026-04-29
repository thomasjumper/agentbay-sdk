FROM python:3.12-slim

WORKDIR /app

# Install agentbay
COPY . .
RUN pip install --no-cache-dir .

# Create data directory for SQLite
RUN mkdir -p /data

# Environment variables
ENV AGENTBAY_DB_PATH=/data/agentbay.db
ENV AGENTBAY_HOST=0.0.0.0
ENV AGENTBAY_PORT=8787

# Expose the local API server
EXPOSE 8787

# Run the local server
CMD ["python", "-m", "agentbay.server"]
