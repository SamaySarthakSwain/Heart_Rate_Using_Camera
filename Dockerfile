# Use the official Python 3.10 slim image for a smaller footprint
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
# (OpenCV requires libgl1 and libglib2.0 to handle image processing correctly on Linux)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the Docker image smaller by not caching the downloaded pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including our_project, external, website, etc.) into the container
COPY . .

# Set PYTHONPATH so Python knows where to find the 'our_project' module
ENV PYTHONPATH=/app

# Hugging Face Spaces automatically routes traffic to port 7860
EXPOSE 7860

# Command to run the FastAPI server on port 7860
CMD ["uvicorn", "our_project.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
