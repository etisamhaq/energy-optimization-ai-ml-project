.PHONY: help setup install install-dl clean test test-file lint format collect-data collect-real preprocess train train-single predict run-api run-dashboard run-mlflow docker-build docker-up docker-down docker-logs pipeline dev deploy

# Default target
help:
	@echo "Energy Optimization ML - Available commands:"
	@echo "  make setup          - Set up development environment"
	@echo "  make install        - Install dependencies"
	@echo "  make install-dl     - Install optional deep-learning dependencies (LSTM)"
	@echo "  make clean          - Clean generated files"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code"
	@echo "  make collect-data   - Generate synthetic demo data"
	@echo "  make collect-real   - Download real BDG2 meter data + Open-Meteo weather"
	@echo "  make preprocess     - Preprocess collected data"
	@echo "  make train          - Train models"
	@echo "  make run-api        - Run FastAPI server"
	@echo "  make run-dashboard  - Run Streamlit dashboard"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start Docker containers"
	@echo "  make docker-down    - Stop Docker containers"

# Setup development environment
setup:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # On Linux/Mac"
	@echo "  venv\\Scripts\\activate     # On Windows"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt -c constraints.txt
	pip install -e .

install-dl:
	pip install -r requirements-dl.txt

# Clean generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist
	rm -rf htmlcov .coverage
	rm -rf .pytest_cache

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run specific test file
test-file:
	@read -p "Enter test file name (e.g., test_models): " file; \
	pytest tests/$$file.py -v

# Lint code
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	pylint src/ tests/ --disable=C0111,C0103,R0903

# Format code
format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black

# Data pipeline
collect-data:
	python src/data/collect_data.py --source synthetic --n-buildings 10 --n-days 365

collect-real:
	python src/data/collect_data.py --source bdg2 --site Rat --n-buildings 10

preprocess:
	python src/data/preprocess.py

# Model training
train:
	python src/models/train.py --model ensemble

train-single:
	@read -p "Enter model name (xgboost/lightgbm/random_forest): " model; \
	python src/models/train.py --model $$model

# Predictions
predict:
	python src/models/predict.py --model ensemble

# Run services
run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run src/visualization/dashboard.py

run-mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Full pipeline
pipeline: collect-data preprocess train
	@echo "Full pipeline completed!"

# Development workflow
dev: install collect-data preprocess
	@echo "Development environment ready!"

# Production deployment
deploy: docker-build docker-up
	@echo "Application deployed!"
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "MLflow: http://localhost:5000"
