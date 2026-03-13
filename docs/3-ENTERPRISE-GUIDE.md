# Isartor Enterprise Guide: Kubernetes & Distributed Scaling

## Why Enterprise Mode?

Enterprise Mode enables horizontal scaling of network I/O independently from GPU compute, and provides shared distributed caching for multi-replica deployments. This is essential for:
- High-throughput, stateless firewall pods
- Efficient GPU utilization via remote inference pools
- Consistent cache hits across replicas (via Redis)

## Layer 1a: Configuring Redis for Exact Match Cache

Switch the exact_cache provider from memory to redis in your `isartor.yaml`:

```yaml
exact_cache:
  provider: redis
  redis_url: "redis://redis-cluster.svc:6379"
  # Optional: redis_db: 0
```

## Layer 2: Configuring vLLM Sidecar for SLM Routing

Switch the slm_router provider from embedded to remote_http:

```yaml
slm_router:
  provider: remote_http
  remote_url: "http://vllm-openai.svc:8000"
  model: "meta-llama/Llama-3-8B-Instruct"
```

### Example: Docker Compose Sidecar Setup

```yaml
docker-compose.yml:

services:
  isartor:
    image: isartor-ai/isartor:latest
    ports:
      - "8080:8080"
    environment:
      - ISARTOR__CACHE_BACKEND=redis
      - ISARTOR__REDIS_URL=redis://redis-cluster:6379
      - ISARTOR__ROUTER_BACKEND=vllm
      - ISARTOR__VLLM_URL=http://vllm-openai:8000
      - ISARTOR__VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
    depends_on:
      - redis
      - vllm-openai

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  vllm-openai:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
```

## Kubernetes Topology: Ideal Enterprise Setup

- **Isartor Deployment:** Stateless pods behind an Ingress controller (NGINX/Traefik).
- **Redis StatefulSet:** Internal distributed cache, accessible only within the cluster.
- **vLLM GPU Deployment:** Dedicated GPU nodes running vLLM, exposed via ClusterIP service.

```
[Ingress]
   |
[Isartor Deployment] <--> [Redis StatefulSet]
   |
   +--> [vLLM Deployment (GPU nodes)]
```

- Isartor pods scale horizontally for network I/O and cache hits.
- Redis ensures cache consistency across all pods.
- vLLM GPU pool scales independently for inference throughput.

---

For advanced configuration, see [README.md](../README.md) and [docs/2-ARCHITECTURE.md](2-ARCHITECTURE.md).
