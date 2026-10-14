# Energy Optimization ML - Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Considerations](#production-considerations)
6. [Monitoring & Maintenance](#monitoring--maintenance)

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- 8GB RAM minimum (16GB recommended for training)
- 10GB free disk space

## Local Development

### 1. Clone the Repository
```bash
git clone https://github.com/etisamhaq/energy-optimization-ai-ml-project.git
cd energy-optimization-ai-ml-project
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Prepare Data
```bash
# Collect initial data
python src/data/collect_data.py --source all

# Preprocess data
python src/data/preprocess.py
```

### 4. Train Models
```bash
# Train ensemble model
python src/models/train.py --model ensemble

# Or train specific model with optimization
python src/models/train.py --model xgboost --optimize
```

### 5. Run Services
```bash
# Run API server
uvicorn src.api.main:app --reload

# Run dashboard (in another terminal)
streamlit run src/visualization/dashboard.py

# Run MLflow UI (optional)
mlflow ui
```

## Docker Deployment

### 1. Build Images
```bash
docker-compose build
```

### 2. Start Services
```bash
docker-compose up -d
```

### 3. Access Services
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

### 4. View Logs
```bash
docker-compose logs -f
```

### 5. Stop Services
```bash
docker-compose down
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup
```bash
# Launch EC2 instance (recommended: t3.large or better)
# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Clone repository
git clone https://github.com/etisamhaq/energy-optimization-ai-ml-project.git
cd energy-optimization-ai-ml-project

# Run Docker deployment
docker-compose up -d
```

#### 2. Configure Security Groups
- Port 8000: API access
- Port 8501: Dashboard access
- Port 22: SSH access

### Google Cloud Platform

#### 1. Using Cloud Run
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/energy-optimization-api

# Deploy API
gcloud run deploy energy-optimization-api \
  --image gcr.io/PROJECT_ID/energy-optimization-api \
  --platform managed \
  --port 8000 \
  --memory 2Gi
```

### Azure Deployment

#### 1. Using Container Instances
```bash
# Create resource group
az group create --name energy-ml-rg --location eastus

# Create container instance
az container create \
  --resource-group energy-ml-rg \
  --name energy-optimization-api \
  --image youracr.azurecr.io/energy-optimization-api \
  --cpu 2 --memory 4 \
  --port 8000
```

## Production Considerations

### 1. Environment Variables
Create a `.env` file for production:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database (if using external DB)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### 2. SSL/TLS Configuration
Use a reverse proxy (nginx) for SSL:
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Database Setup
For production, use PostgreSQL:
```sql
CREATE DATABASE energy_optimization;
CREATE USER energy_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE energy_optimization TO energy_user;
```

### 4. Scaling Considerations
- Use Redis for caching predictions
- Implement load balancing for multiple API instances
- Use S3 or similar for model storage
- Set up auto-scaling based on CPU/memory usage

## Monitoring & Maintenance

### 1. Health Checks
```bash
# API health check
curl http://localhost:8000/health
```

### 2. Log Management
```bash
# View API logs
docker logs energy-optimization-api -f

# Export logs
docker logs energy-optimization-api > api_logs_$(date +%Y%m%d).log
```

### 3. Backup Strategy
```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup data
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/processed/
```

### 4. Model Updates
```bash
# Retrain models with new data
python src/models/train.py --model ensemble

# Deploy new model
docker-compose restart api
```

### 5. Performance Monitoring
- Set up Prometheus + Grafana for metrics
- Monitor API response times
- Track prediction accuracy over time
- Set up alerts for anomalies

## Troubleshooting

### Common Issues

1. **Out of Memory During Training**
   - Reduce batch size in model parameters
   - Use fewer models in ensemble
   - Increase system memory

2. **Slow API Response**
   - Enable model caching
   - Reduce model complexity
   - Use async endpoints

3. **Docker Build Failures**
   - Clear Docker cache: `docker system prune -a`
   - Check disk space
   - Verify internet connectivity

### Support
- GitHub Issues: https://github.com/etisamhaq/energy-optimization-ai-ml-project/issues
- Email: energy-ml@example.com