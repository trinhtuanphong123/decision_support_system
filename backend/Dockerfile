# ====================================
# Backend API - NYC Airbnb Predictor
# ====================================

FROM python:3.11-slim-bookworm

# Metadata
LABEL maintainer="your.email@example.com"
LABEL service="airbnb-backend-api"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY predict.py .
COPY model.bin .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-9696}/health || exit 1

# Run the application
CMD ["python", "predict.py"]