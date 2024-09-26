# Stage 1: Build stage
FROM --platform=linux/amd64 python:3.10-slim as builder

# Set the working directory
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install necessary Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final stage
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files from the builder stage
COPY --from=builder /app /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Ensure uvicorn is installed in the final image
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "websocket:app", "--host", "0.0.0.0", "--port", "8000"]
