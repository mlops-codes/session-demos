#!/bin/bash

echo "ML Model Monitoring Setup Script"
echo "================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"

# Create necessary directories
echo "Creating directories..."
mkdir -p logs

# Build and start services
echo "Building and starting services..."
docker-compose up --build -d

echo "Waiting for services to be ready..."
sleep 30

# Check if services are running
echo "Checking service status..."
docker-compose ps

echo ""
echo "Setup complete! Services are now running:"
echo "- ML Model API: http://localhost:8000"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin123)"
echo ""
echo "Run ./scripts/test_api.sh to test the ML model API"
echo "Run ./scripts/stop.sh to stop all services"