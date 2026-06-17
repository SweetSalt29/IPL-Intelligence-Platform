FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (src, data, config, dashboard)
COPY . .

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the API and Dashboard port
EXPOSE 8001

# Run the FastAPI server using Uvicorn directly for better performance
CMD ["python", "-m", "uvicorn", "src.models.serve:app", "--host", "0.0.0.0", "--port", "8001"]
