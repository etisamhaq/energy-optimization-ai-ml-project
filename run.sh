#!/bin/bash

# Energy Optimization ML - Run Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main menu
show_menu() {
    echo "=================================="
    echo "Energy Optimization ML - Run Menu"
    echo "=================================="
    echo "1. Setup Development Environment"
    echo "2. Collect & Preprocess Data"
    echo "3. Train Models"
    echo "4. Run API Server"
    echo "5. Run Dashboard"
    echo "6. Run Tests"
    echo "7. Run Full Pipeline"
    echo "8. Docker Deploy"
    echo "0. Exit"
    echo "=================================="
}

# Setup environment
setup_environment() {
    print_status "Setting up development environment..."
    
    if [ ! -d "venv" ]; then
        python -m venv venv
        print_status "Virtual environment created"
    fi
    
    source venv/bin/activate || source venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
    
    print_status "Environment setup complete!"
}

# Data collection and preprocessing
collect_and_preprocess() {
    print_status "Collecting and preprocessing data..."
    
    python src/data/collect_data.py --source all --n-buildings 10 --n-days 365
    print_status "Data collection complete"
    
    python src/data/preprocess.py
    print_status "Data preprocessing complete"
}

# Train models
train_models() {
    print_status "Training models..."
    
    echo "Select model to train:"
    echo "1. Ensemble (all models)"
    echo "2. XGBoost"
    echo "3. LightGBM"
    echo "4. Random Forest"
    echo "5. Prophet"
    
    read -p "Enter choice: " model_choice
    
    case $model_choice in
        1) python src/models/train.py --model ensemble ;;
        2) python src/models/train.py --model xgboost --optimize ;;
        3) python src/models/train.py --model lightgbm --optimize ;;
        4) python src/models/train.py --model random_forest --optimize ;;
        5) python src/models/train.py --model prophet ;;
        *) print_error "Invalid choice" ;;
    esac
}

# Run API server
run_api() {
    print_status "Starting API server..."
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
}

# Run dashboard
run_dashboard() {
    print_status "Starting Streamlit dashboard..."
    streamlit run src/visualization/dashboard.py
}

# Run tests
run_tests() {
    print_status "Running tests..."
    pytest tests/ -v --cov=src --cov-report=html
    print_status "Tests complete. Coverage report available in htmlcov/index.html"
}

# Run full pipeline
run_pipeline() {
    print_status "Running full pipeline..."
    collect_and_preprocess
    python src/models/train.py --model ensemble
    print_status "Pipeline complete!"
}

# Docker deployment
docker_deploy() {
    if ! command_exists docker; then
        print_error "Docker is not installed"
        return 1
    fi
    
    print_status "Building and deploying with Docker..."
    docker-compose build
    docker-compose up -d
    
    print_status "Deployment complete!"
    echo "API: http://localhost:8000"
    echo "Dashboard: http://localhost:8501"
    echo "MLflow: http://localhost:5000"
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice: " choice
    
    case $choice in
        1) setup_environment ;;
        2) collect_and_preprocess ;;
        3) train_models ;;
        4) run_api ;;
        5) run_dashboard ;;
        6) run_tests ;;
        7) run_pipeline ;;
        8) docker_deploy ;;
        0) 
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
    clear
done