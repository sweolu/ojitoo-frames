#!/bin/bash
set -e  # Exit immediately if a command fails

# ------------------------------
# Configuration
# ------------------------------
APP_NAME="ojitoo-frames"
IMAGE_NAME="ojitoo-frames:latest"
CONTAINER_NAME="ojitoo-frames"
PORT=8000
ENV_FILE=".env"   # Path to your env file

# ------------------------------
# Step 1: Ensure script runs as root or with sudo
# ------------------------------
if [ "$EUID" -ne 0 ]; then
  echo "⚠️  Please run as root (use: sudo ./deploy.sh)"
  exit 1
fi

# ------------------------------
# Step 2: Check Docker installation
# ------------------------------
if ! command -v docker &> /dev/null
then
    echo "🚨 Docker not found. Installing Docker..."
    apt-get update && apt-get install -y ca-certificates curl gnupg lsb-release
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    echo "✅ Docker installed successfully."
fi

# ------------------------------
# Step 3: Optional — Check NVIDIA support
# ------------------------------
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo "🟢 NVIDIA GPU detected. Running container with GPU support."
else
    GPU_FLAG=""
    echo "⚪ No NVIDIA GPU detected. Running without GPU acceleration."
fi

# ------------------------------
# Step 4: Verify .env file exists
# ------------------------------
if [ ! -f "$ENV_FILE" ]; then
  echo "🚨 ERROR: $ENV_FILE not found in the current directory."
  echo "Please create a .env file with the required environment variables."
  exit 1
fi

# ------------------------------
# Step 5: Build the Docker image
# ------------------------------
echo "🛠️  Building Docker image..."
docker build -t $IMAGE_NAME .

# ------------------------------
# Step 6: Stop and remove any old container
# ------------------------------
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🧹 Stopping old container..."
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
fi

# ------------------------------
# Step 7: Run the new container with .env
# ------------------------------
echo "🚀 Starting new container with env vars..."
docker run -d \
  --name $CONTAINER_NAME \
  -p ${PORT}:8000 \
  $GPU_FLAG \
  --env-file $ENV_FILE \
  --restart unless-stopped \
  $IMAGE_NAME

# ------------------------------
# Step 8: Confirm deployment
# ------------------------------
echo "✅ Deployment complete!"
echo "🌐 App running on: http://<your-ip>:${PORT}"
echo "📄 Environment variables loaded from: $ENV_FILE"

