#!/bin/bash

echo "Stopping ML Model Monitoring Services"
echo "====================================="

# Stop and remove containers
docker-compose down

echo "Services stopped successfully!"
echo ""
echo "To completely remove all data:"
echo "  docker-compose down -v  # Remove volumes"
echo "  docker system prune     # Clean up unused resources"