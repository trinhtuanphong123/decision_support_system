# Use Python 3.11 slim version
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /code

# Install pip and update it
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY predict.py model.bin ./

# Expose port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:9696/health')" || exit 1

# Run the application
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]