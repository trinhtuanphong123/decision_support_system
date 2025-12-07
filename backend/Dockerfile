# ====================================
# Backend API - NYC Airbnb Predictor
# Updated with Monitoring & Validation
# ====================================

FROM python:3.11-slim-bookworm

# Metadata
LABEL maintainer="trinhtuanphong123@example.com"
LABEL service="airbnb-backend-api"
LABEL version="2.0.0"
LABEL description="FastAPI backend with ML model, monitoring, and data validation"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
# Order matters: least changing files first (for better caching)
COPY model.bin .
COPY monitoring.py .
COPY data_validator.py .
COPY predict.py .

# Optional: Copy test files for debugging (can be excluded in .dockerignore)
# COPY test/ ./test/

# Create directories for outputs
RUN mkdir -p /app/logs /app/metrics

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Render will override with $PORT)
EXPOSE 9696

# Health check
# Note: Uses $PORT environment variable if available (for Render)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-9696}/health || exit 1

# Environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run the application
# Note: predict.py reads PORT from environment variable
CMD ["python", "predict.py"]