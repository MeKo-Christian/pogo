# Pogo OCR Deployment

This directory contains deployment configurations for the Pogo OCR server.

## Quick Start

```bash
# From the deployment/ directory
docker-compose up --build

# Access your OCR service
curl -s -F image=@../testdata/images/simple_text.png http://localhost:8080/ocr/image | jq
```

## Files

- **Dockerfile** - Multi-stage Docker build for the OCR server
- **docker-compose.yml** - Production-ready Docker Compose configuration
- **nginx.conf** - Reverse proxy configuration for load balancing
- **README.md** - This deployment guide

Note: The `.dockerignore` file is located in the project root since the build context is `..` (parent directory).

## Usage

### Basic Deployment

```bash
cd deployment/
docker-compose up -d
```

### With Reverse Proxy

```bash
cd deployment/
docker-compose --profile proxy up -d
```

This will start both the OCR service and an nginx reverse proxy.

### Custom Configuration

Copy and modify the docker-compose.yml:

```bash
cp docker-compose.yml docker-compose.override.yml
# Edit docker-compose.override.yml with your settings
docker-compose up -d
```

### Environment Variables

Create a `.env` file in the deployment/ directory:

```env
VERSION=v1.0.0
POGO_LOG_LEVEL=debug
POGO_SERVER_PORT=8080
```

### Custom Models

Mount your custom models directory:

```yaml
# In docker-compose.override.yml
services:
  pogo-ocr:
    volumes:
      - ../my-custom-models:/usr/share/pogo/models:ro
```

## Production Considerations

- Resource limits are configured for typical workloads
- Health checks ensure service reliability
- Non-root execution for security
- Restart policies handle failures gracefully

## Monitoring

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f pogo-ocr

# Health check
curl http://localhost:8080/health
```