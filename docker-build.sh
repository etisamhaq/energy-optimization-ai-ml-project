#!/bin/bash

# Build the Docker image
echo "Building Energy Optimization ML Docker image..."

# Clean any previous builds
docker-compose down 2>/dev/null || true

# Build the image
docker-compose build --no-cache api

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can now run: docker-compose up"
else
    echo "Build failed. Please check the error messages above."
    exit 1
fi