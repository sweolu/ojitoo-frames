#!/bin/bash
set -e

echo "Starting deployment..."
cd /opt/ojitoo-frames

echo "Pulling latest code..."
git pull origin main

echo "Stopping current container..."
docker-compose down || true

echo "Building new image..."
docker-compose build --no-cache

echo "Starting container..."
docker-compose up -d

echo "Waiting for service..."
sleep 10

if [ "$(docker inspect -f '{{.State.Running}}' ojitoo-frames)" == "true" ]; then
    echo "✅ Deployment successful!"
    docker-compose logs --tail=50
else
    echo "❌ Deployment failed!"
    docker-compose logs --tail=100
    exit 1
fi
