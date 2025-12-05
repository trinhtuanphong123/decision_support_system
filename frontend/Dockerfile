# ====================================
# Frontend UI - NYC Airbnb Predictor
# ====================================

FROM python:3.11-slim-bookworm

# Metadata
LABEL maintainer="trinhtuanphong@example.com"
LABEL service="airbnb-frontend-ui"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .streamlit directory for config
RUN mkdir -p ~/.streamlit

# Streamlit configuration
# LƯU Ý: Tôi đã XÓA dòng 'port = 8501' ở đây để tránh xung đột
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = '#FF4B4B'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#F0F2F6'\n\
textColor = '#262730'\n\
font = 'sans serif'\n\
" > ~/.streamlit/config.toml

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit default port (Chỉ mang tính thông báo)
EXPOSE 8501

# Health check for Streamlit
# Quan trọng: Dùng biến PORT để healthcheck đúng chỗ
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Run Streamlit app
# SENIOR FIX: Sử dụng sh -c để nhận diện biến môi trường $PORT từ Render
CMD sh -c "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"