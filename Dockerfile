# Multi-stage build for Energy Optimization ML
FROM python:3.10-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt constraints.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements-docker.txt -c constraints.txt

# Production stage
FROM python:3.10-slim-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mluser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder --chown=mluser:mluser /root/.local /home/mluser/.local

# Copy application code
COPY --chown=mluser:mluser . .

# Switch to non-root user
USER mluser

# Add local bin to PATH
ENV PATH=/home/mluser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]