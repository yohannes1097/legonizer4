#!/bin/bash

# Script deployment untuk Legonizer4

set -e

echo "🚀 Legonizer4 Deployment Script"
echo "================================"

# Configuration
DOCKER_IMAGE="legonizer4:latest"
CONTAINER_NAME="legonizer4-api"
API_PORT="8000"
DATA_DIR="./data"
CONFIG_FILE="./config.json"

# Functions
check_requirements() {
    echo "📋 Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker (optional)
    if command -v nvidia-docker &> /dev/null; then
        echo "✅ NVIDIA Docker detected - GPU support available"
    else
        echo "⚠️  NVIDIA Docker not found - CPU only mode"
    fi
    
    echo "✅ Requirements check passed"
}

build_image() {
    echo "🔨 Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    echo "✅ Docker image built successfully"
}

setup_data_directories() {
    echo "📁 Setting up data directories..."
    
    mkdir -p $DATA_DIR/raw
    mkdir -p $DATA_DIR/processed
    mkdir -p $DATA_DIR/models
    mkdir -p $DATA_DIR/metrics/reports
    
    echo "✅ Data directories created"
}

create_config() {
    echo "⚙️  Creating configuration..."
    
    if [ ! -f $CONFIG_FILE ]; then
        echo "Creating default config.json..."
        cat > $CONFIG_FILE << EOF
{
  "model": {
    "num_classes": 10,
    "pretrained": true,
    "feature_dim": 1792,
    "dropout_rate": 0.5
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.0001,
    "val_split": 0.2,
    "early_stopping_patience": 5,
    "seed": 42,
    "device": "auto",
    "augment": true
  },
  "preprocessing": {
    "min_contour_area": 1000,
    "blur_kernel": 5,
    "target_size": [224, 224],
    "save_visualization": false,
    "max_workers": 4
  },
  "search": {
    "index_type": "L2",
    "nlist": 100,
    "nprobe": 10,
    "dimension": 1792
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false,
    "workers": 1,
    "log_level": "info"
  },
  "data_dir": "data",
  "raw_data_dir": "data/raw",
  "processed_data_dir": "data/processed",
  "models_dir": "data/models",
  "metrics_dir": "data/metrics"
}
EOF
        echo "✅ Default config.json created"
    else
        echo "✅ Config.json already exists"
    fi
}

deploy_with_docker_compose() {
    echo "🐳 Deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose down
    
    # Start services
    docker-compose up -d
    
    echo "✅ Services started"
    echo "📊 Checking service status..."
    docker-compose ps
}

deploy_standalone() {
    echo "🐳 Deploying standalone container..."
    
    # Stop existing container
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name $CONTAINER_NAME \
        -p $API_PORT:8000 \
        -v $(pwd)/$DATA_DIR:/app/data \
        -v $(pwd)/$CONFIG_FILE:/app/config.json \
        --restart unless-stopped \
        $DOCKER_IMAGE
    
    echo "✅ Container started"
}

wait_for_api() {
    echo "⏳ Waiting for API to be ready..."
    
    for i in {1..30}; do
        if curl -f http://localhost:$API_PORT/health &>/dev/null; then
            echo "✅ API is ready!"
            return 0
        fi
        echo "Waiting... ($i/30)"
        sleep 2
    done
    
    echo "❌ API failed to start"
    return 1
}

show_status() {
    echo "📊 Deployment Status"
    echo "===================="
    echo "API URL: http://localhost:$API_PORT"
    echo "Health Check: http://localhost:$API_PORT/health"
    echo "API Docs: http://localhost:$API_PORT/docs"
    echo ""
    echo "Container Status:"
    docker ps --filter name=$CONTAINER_NAME
}

# Main deployment flow
main() {
    local deployment_type=${1:-"compose"}
    
    echo "Starting deployment (type: $deployment_type)..."
    
    check_requirements
    setup_data_directories
    create_config
    build_image
    
    case $deployment_type in
        "compose")
            deploy_with_docker_compose
            ;;
        "standalone")
            deploy_standalone
            ;;
        *)
            echo "❌ Invalid deployment type: $deployment_type"
            echo "Usage: $0 [compose|standalone]"
            exit 1
            ;;
    esac
    
    wait_for_api
    show_status
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "You can now use the API at http://localhost:$API_PORT"
}

# Run main function with arguments
main "$@"
