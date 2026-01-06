# Monitoring & MLOps Stack Setup Guide

## Overview
This document describes the complete setup of a production-ready monitoring and MLOps infrastructure for the Credit Risk Engine API using Docker Compose.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Docker Compose Stack                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │   API    │───▶│ MLflow   │───▶│ MinIO    │                  │
│  │  :8000   │    │  :5000   │    │  :9000   │ (S3 Artifacts)   │
│  └────┬─────┘    └────┬─────┘    └──────────┘                  │
│       │               │                                           │
│       │               │           ┌──────────┐                  │
│       │               └──────────▶│ Postgres │                  │
│       │                           │  :5432   │ (Metadata)       │
│       │                           └──────────┘                  │
│       │                                                           │
│       │ /metrics                                                 │
│       ▼                                                           │
│  ┌──────────┐    ┌──────────┐                                  │
│  │Prometheus│───▶│ Grafana  │                                  │
│  │  :9090   │    │  :3000   │                                  │
│  └──────────┘    └──────────┘                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Components & Purpose

### 1. **PostgreSQL (Postgres:14)**
**Purpose:** Backend database for MLflow tracking server

**What it stores:**
- Experiment metadata
- Run parameters and metrics
- Model versions
- Tags and notes

**Why we need it:**
- Persistent storage for all ML experiment tracking data
- Better than file-based backend for production
- Supports concurrent access from multiple MLflow clients

**Configuration:**
```yaml
postgres:
  image: postgres:14
  environment:
    POSTGRES_USER: mlflow
    POSTGRES_PASSWORD: mlflow
    POSTGRES_DB: mlflow
  ports:
    - "5432:5432"
  volumes:
    - pgdata:/var/lib/postgresql/data
```

---

### 2. **MinIO**
**Purpose:** S3-compatible object storage for ML artifacts

**What it stores:**
- Model binaries (.pkl files)
- Training plots and visualizations
- Large datasets
- SHAP explainer objects
- Any artifact larger than what fits in Postgres

**Why we need it:**
- MLflow needs separate storage for large files
- S3-compatible API (works with boto3)
- Self-hosted alternative to AWS S3
- Better than local filesystem for distributed setups

**Configuration:**
```yaml
minio:
  image: minio/minio
  command: server /data --console-address ":9001"
  environment:
    MINIO_ROOT_USER: minio
    MINIO_ROOT_PASSWORD: minio123
  ports:
    - "9000:9000"  # S3 API
    - "9001:9001"  # Web Console
  volumes:
    - minio:/data
```

**Accessing MinIO Console:**
- URL: http://localhost:9001
- Login: minio / minio123

---

### 3. **MLflow (Custom Image)**
**Purpose:** ML experiment tracking and model registry

**What it does:**
- Tracks experiments (parameters, metrics, artifacts)
- Stores model versions
- Provides model registry for deployment
- Serves models via REST API (optional)
- Web UI for visualizing experiments

**Why we need it:**
- Reproducibility: Track every model training run
- Comparison: Compare different model versions
- Collaboration: Team can see all experiments
- Deployment: Model registry for production models

**Custom Dockerfile (`Dockerfile.mlflow`):**
```dockerfile
FROM ghcr.io/mlflow/mlflow:v2.9.2

# Install PostgreSQL adapter and boto3 for S3
RUN pip install --no-cache-dir psycopg2-binary boto3

# Expose port
EXPOSE 5000

# Override entrypoint to ensure 0.0.0.0 binding
ENTRYPOINT []
CMD ["sh", "-c", "mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} --host 0.0.0.0 --port 5000"]
```

**Why custom image?**
- Official image lacks `psycopg2` for Postgres connectivity
- Official image lacks `boto3` for S3/MinIO integration
- Need to force binding to `0.0.0.0` instead of `127.0.0.1`

**Configuration:**
```yaml
mlflow:
  build:
    context: .
    dockerfile: Dockerfile.mlflow
  environment:
    MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
    MLFLOW_DEFAULT_ARTIFACT_ROOT: s3://mlflow/
    MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    AWS_ACCESS_KEY_ID: minio
    AWS_SECRET_ACCESS_KEY: minio123
  ports:
    - "5000:5000"
```

**Accessing MLflow UI:**
- URL: http://localhost:5000
- No login required (configure authentication for production)

---

### 4. **Prometheus (v2.49.1)**
**Purpose:** Metrics collection and time-series database

**What it does:**
- Scrapes metrics from API `/metrics` endpoint
- Stores time-series data (requests, latency, errors)
- Provides query language (PromQL)
- Alerts on metric thresholds

**Why we need it:**
- Monitor API performance in real-time
- Track request rates, error rates, latency
- Historical data for trend analysis
- Foundation for alerting

**Configuration (`prometheus.yml`):**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000']
```

**Key Metrics Collected:**
- `api_requests_total` - Counter of all API requests by endpoint and status
- `api_request_latency_seconds` - Histogram of request latencies by endpoint
- `python_gc_*` - Python garbage collection metrics
- `process_*` - CPU, memory, file descriptors

**Accessing Prometheus UI:**
- URL: http://localhost:9090
- Query examples:
  - `rate(api_requests_total[5m])` - Request rate per second
  - `histogram_quantile(0.95, api_request_latency_seconds)` - 95th percentile latency

---

### 5. **Grafana (v10.2.3)**
**Purpose:** Visualization and dashboarding

**What it does:**
- Creates dashboards from Prometheus metrics
- Visualizes trends, patterns, anomalies
- Sets up alerts
- Provides user-friendly UI for stakeholders

**Why we need it:**
- Prometheus UI is basic, Grafana is feature-rich
- Pre-built dashboard templates
- Better for non-technical stakeholders
- Supports multiple data sources

**Configuration:**
```yaml
grafana:
  image: grafana/grafana:10.2.3
  environment:
    GF_SECURITY_ADMIN_USER: admin
    GF_SECURITY_ADMIN_PASSWORD: admin
  ports:
    - "3000:3000"
  volumes:
    - grafanadata:/var/lib/grafana
```

**Accessing Grafana:**
- URL: http://localhost:3000
- Login: admin / admin (change on first login)

**Setup Steps:**
1. Add Prometheus data source: http://prometheus:9090
2. Import dashboard or create custom
3. Create panels for key metrics

---

### 6. **FastAPI Application**
**Purpose:** Credit risk prediction API with instrumentation

**What we added:**
- **Prometheus metrics endpoint** at `/metrics`
- **Request counters** - Track success/error counts per endpoint
- **Latency histograms** - Measure response times

**Implementation (`api/main.py`):**
```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from .routes import router

app = FastAPI(title="Credit Risk Engine API")

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(router)
```

**Instrumented Routes (`api/routes.py`):**
```python
from prometheus_client import Counter, Histogram
import time

# Metrics definition
REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Total API requests",
    labelnames=["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    labelnames=["endpoint"],
)

@router.post("/predict")
def predict_credit_risk(request: PredictionRequest):
    start = time.perf_counter()
    try:
        # ... prediction logic ...
        REQUEST_COUNTER.labels(endpoint="predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - start)
        return result
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="predict", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - start)
        raise
```

**Added Dependency:**
```
prometheus_client==0.20.0
```

---

## Implementation Steps

### Step 1: Created Docker Compose Stack
Added all services to `docker-compose.yml` with proper networking and dependencies.

### Step 2: Created Prometheus Configuration
Created `prometheus.yml` to scrape metrics from API and self-monitor.

### Step 3: Built Custom MLflow Image
Created `Dockerfile.mlflow` to:
- Install missing dependencies (psycopg2-binary, boto3)
- Force binding to 0.0.0.0 for external access

### Step 4: Instrumented API
- Added `prometheus_client` to requirements
- Created metrics (Counter, Histogram)
- Instrumented `/predict` and `/explain` endpoints
- Mounted `/metrics` endpoint

### Step 5: Deployed Stack
```bash
docker-compose build
docker-compose up -d
```

### Step 6: Verified All Services
```bash
# Check all containers running
docker-compose ps

# Test MLflow
curl http://localhost:5000/health

# Test API metrics
curl http://localhost:8000/metrics

# Test Prometheus
curl http://localhost:9090/-/healthy

# Test Grafana
curl http://localhost:3000/api/health
```

---

## Troubleshooting: The MLflow Binding Issue

### Problem
MLflow container kept crashing with errors:
1. `ModuleNotFoundError: No module named 'psycopg2'`
2. After fixing deps, MLflow bound to `127.0.0.1:5000` instead of `0.0.0.0:5000`

### Root Causes

#### Issue 1: Missing Dependencies
Official `ghcr.io/mlflow/mlflow:v2.9.2` image doesn't include:
- `psycopg2-binary` - PostgreSQL Python adapter
- `boto3` - AWS SDK for S3/MinIO integration

**Solution:** Created custom Dockerfile to install dependencies.

#### Issue 2: Incorrect Binding (127.0.0.1 vs 0.0.0.0)

**What's the difference?**

| Address | Scope | Docker Port Mapping | Use Case |
|---------|-------|---------------------|----------|
| `127.0.0.1` | Loopback only | ❌ Doesn't work | Internal container communication only |
| `0.0.0.0` | All interfaces | ✅ Works | Accessible from host and external networks |

**Why it matters in Docker:**
```
Docker Port Flow:
Host browser → localhost:5000 → Docker bridge → Container:5000

If container binds to 127.0.0.1:
Host → Docker bridge → ❌ Can't reach 127.0.0.1 (loopback is container-internal)

If container binds to 0.0.0.0:
Host → Docker bridge → ✅ Reaches all interfaces including bridge network
```

**Why MLflow defaulted to 127.0.0.1:**
The `mlflow server --host 0.0.0.0` flag sets the Flask app host, but gunicorn (the WSGI server) ignores it and defaults to `127.0.0.1`.

**Solution:** 
Override the CMD in Dockerfile to explicitly control the command:
```dockerfile
CMD ["sh", "-c", "mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} --host 0.0.0.0 --port 5000"]
```

This forces the shell to expand environment variables and ensures the correct binding.

---

## How to Use the Stack

### Training a Model with MLflow Tracking

```python
import mlflow
import mlflow.lightgbm

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit-risk-model")

# Start run
with mlflow.start_run(run_name="lgb_v1"):
    # Log parameters
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("auc", 0.92)
    
    # Log model
    mlflow.lightgbm.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("shap_summary.png")
    mlflow.log_artifact("confusion_matrix.png")
```

### Monitoring API Performance

**View Prometheus Metrics:**
1. Open http://localhost:9090
2. Query: `rate(api_requests_total{endpoint="predict"}[5m])`
3. Graph shows requests per second

**Create Grafana Dashboard:**
1. Open http://localhost:3000, login (admin/admin)
2. Add Prometheus data source: http://prometheus:9090
3. Create dashboard with panels:
   - Request Rate: `rate(api_requests_total[5m])`
   - Error Rate: `rate(api_requests_total{status="error"}[5m])`
   - P95 Latency: `histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))`
   - Success Rate: `sum(rate(api_requests_total{status="success"}[5m])) / sum(rate(api_requests_total[5m]))`

### Comparing Model Runs in MLflow

1. Open http://localhost:5000
2. Navigate to experiment
3. Select multiple runs
4. Click "Compare"
5. View parameter/metric differences side-by-side

---

## Files Created/Modified

### New Files
- `docker-compose.yml` - Orchestrates all services
- `Dockerfile.mlflow` - Custom MLflow image with dependencies
- `prometheus.yml` - Prometheus scrape configuration
- `docs/monitoring-stack-setup.md` - This document

### Modified Files
- `api/main.py` - Added metrics endpoint mount
- `api/routes.py` - Added Prometheus instrumentation
- `requirements.txt` - Added prometheus_client

---

## Port Reference

| Service | Port | Purpose | Access |
|---------|------|---------|--------|
| API | 8000 | HTTP API + /metrics | http://localhost:8000 |
| MLflow | 5000 | Tracking UI & API | http://localhost:5000 |
| Prometheus | 9090 | Metrics DB & Query UI | http://localhost:9090 |
| Grafana | 3000 | Dashboard UI | http://localhost:3000 |
| Postgres | 5432 | Database | Internal only |
| MinIO S3 API | 9000 | Object storage | Internal only |
| MinIO Console | 9001 | Web UI | http://localhost:9001 |

---

## Volume Reference

| Volume | Purpose | Contains |
|--------|---------|----------|
| `pgdata` | PostgreSQL data | Experiment metadata, run history |
| `minio` | MinIO data | Model artifacts, plots, large files |
| `promdata` | Prometheus data | Time-series metrics |
| `grafanadata` | Grafana data | Dashboard definitions, settings |

**Backup Strategy:**
```bash
# Backup Postgres
docker exec credit-risk-engine-postgres-1 pg_dump -U mlflow mlflow > mlflow_backup.sql

# Backup MinIO (using mc client)
mc mirror local/minio/mlflow /backup/mlflow-artifacts

# Backup Prometheus
docker cp credit-risk-engine-prometheus-1:/prometheus /backup/prometheus-data
```

---

## Production Considerations

### Security
- [ ] Change default passwords
- [ ] Enable SSL/TLS for all services
- [ ] Implement authentication for MLflow UI
- [ ] Use secrets management (Vault, AWS Secrets Manager)
- [ ] Enable Grafana SSO
- [ ] Restrict network access with firewalls

### High Availability
- [ ] Run multiple MLflow replicas behind load balancer
- [ ] Use managed Postgres (RDS) with replication
- [ ] Use AWS S3 instead of MinIO
- [ ] Add Prometheus remote storage (Thanos)
- [ ] Configure Grafana with external database

### Monitoring the Monitoring
- [ ] Set up Prometheus alerts for service health
- [ ] Monitor disk usage for volumes
- [ ] Alert on Postgres/MinIO connection failures
- [ ] Track MLflow API response times

### Resource Management
- [ ] Add CPU/memory limits to all containers
- [ ] Configure log rotation
- [ ] Set retention policies (Prometheus, MLflow)
- [ ] Monitor container resource usage

### Disaster Recovery
- [ ] Automated daily backups
- [ ] Test restore procedures
- [ ] Document recovery steps
- [ ] Offsite backup storage

---

## Common Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f mlflow
docker-compose logs -f api

# Restart a service
docker-compose restart mlflow

# Rebuild after code changes
docker-compose build api
docker-compose up -d api

# Check service health
docker-compose ps
docker-compose logs --tail=20 mlflow

# Access container shell
docker exec -it credit-risk-engine-mlflow-1 sh

# Check network connectivity
docker exec credit-risk-engine-api-1 wget -O- http://mlflow:5000/health

# Remove everything (including volumes)
docker-compose down -v
```

---

## Next Steps

1. **Set up Grafana dashboards**
   - Import pre-built dashboards
   - Create custom panels for business metrics

2. **Configure alerts**
   - Prometheus alert rules for high error rates
   - Grafana notifications (email, Slack)

3. **Integrate MLflow with training pipeline**
   - Update training scripts to log to MLflow
   - Create model registry workflow

4. **Implement CI/CD**
   - Automate model deployment from MLflow registry
   - Add API deployment pipeline

5. **Add more metrics**
   - Business metrics (approval rate, average loan amount)
   - Model performance metrics (drift detection)
   - Infrastructure metrics (database connections, memory)

6. **Document runbooks**
   - Incident response procedures
   - Common troubleshooting steps
   - Escalation paths

---

## Key Learnings

### Docker Networking
- Containers use service names as DNS (e.g., `http://postgres:5432`)
- Port mapping requires binding to `0.0.0.0`, not `127.0.0.1`
- Internal communication doesn't need port mapping

### MLflow Best Practices
- Always use backend database (Postgres) for production
- Use object storage (S3/MinIO) for artifacts
- Custom Docker images often needed for dependencies
- Test connectivity before deploying

### Observability
- Three pillars: Metrics (Prometheus), Logs (future), Traces (future)
- Instrument critical paths (predict, explain endpoints)
- Use labels for filtering (endpoint, status)
- Histograms for latency, Counters for events

### Docker Compose Tips
- Use `depends_on` for startup order
- Mount configs as read-only volumes
- Use named volumes for persistence
- Always specify restart policies

---

## Conclusion

This stack provides:
✅ **Experiment tracking** - MLflow with Postgres + MinIO  
✅ **Real-time monitoring** - Prometheus scraping API metrics  
✅ **Visualization** - Grafana dashboards  
✅ **Production-ready foundation** - Containerized, scalable architecture  

The setup is suitable for development and small-scale production. For larger deployments, consider managed services (RDS, S3, CloudWatch) and Kubernetes orchestration.
