# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Clean up
RUN rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip install torch torchvision torchaudio

# Set the default command to check CUDA availability
CMD ["nvidia-smi", "&&", "python3", "-c", "import torch; print(torch.cuda.is_available())"]
