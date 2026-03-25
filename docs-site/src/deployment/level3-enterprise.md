# Level 3 — Enterprise Deployment

> **Fully decoupled microservices: stateless firewall pods + auto-scaling GPU inference pools.**

This guide covers deploying Isartor on Kubernetes with Helm, horizontal pod autoscaling, dedicated GPU inference pools (vLLM or TGI), service mesh integration, and production-grade observability.

---

## When to Use Level 3

| ✅ Good Fit | ❌ Overkill For |
| --- | --- |
| 100+ concurrent users | < 50 users → Level 2 Docker Compose |
| Multi-region / multi-zone HA | Single-machine development → Level 1 |
| Auto-scaling GPU inference | No GPU budget → Level 1 embedded candle |
| Compliance: mTLS, audit logs, RBAC | Hobby projects / PoCs |
| Cost optimisation via scale-to-zero | Teams without Kubernetes experience |

---

## Architecture

```text
                        ┌────────────────────┐
                        │    Ingress / ALB    │
                        │  (TLS termination)  │
                        └──────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      Firewall Deployment     │
                    │      (N stateless pods)       │
                    │                              │
                    │  ┌────────┐   ┌────────┐    │
                    │  │ Pod 1  │   │ Pod N  │    │
                    │  │isartor │   │isartor │    │
                    │  └────────┘   └────────┘    │
                    │                              │
                    │  HPA: CPU / custom metrics   │
                    └──────────────┬───────────────┘
                                   │
                          Internal ClusterIP
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼───────┐  ┌────────▼───────┐   ┌────────▼───────┐
  │ Inference Pool  │  │ Embedding Pool  │   │ Cloud LLM      │
  │ (vLLM / TGI)   │  │ (TEI / llama)   │   │ (OpenAI / etc) │
  │                 │  │                 │   │ (Layer 3 only)  │
  │ GPU Nodes       │  │ CPU/GPU Nodes   │   └────────────────┘
  │ HPA on GPU util │  │ HPA on RPS      │
  └─────────────────┘  └─────────────────┘
```

### Component Summary

| Component | Replicas | Scaling Metric | Resource |
| --- | --- | --- | --- |
| **Firewall** | 2–20 | CPU utilisation / request rate | CPU nodes |
| **Inference Pool** (vLLM) | 1–N | GPU utilisation / queue depth | GPU nodes |
| **Embedding Pool** (TEI) | 1–N | Requests per second | CPU or GPU nodes (optional; default uses in-process candle) |
| **OTel Collector** | 1 (DaemonSet or Deployment) | — | CPU nodes |
| **Ingress Controller** | 1–2 | — | CPU nodes |

---

## Prerequisites

| Requirement | Details |
| --- | --- |
| **Kubernetes cluster** | 1.28+ (EKS, GKE, AKS, or bare metal) |
| **Helm** | v3.12+ |
| **kubectl** | Matching cluster version |
| **GPU nodes** (for inference pool) | NVIDIA GPU Operator installed, or GKE/EKS GPU node pools |
| **Container registry** | For pushing the Isartor firewall image |
| **Ingress controller** | nginx-ingress, Istio, or cloud ALB |

---

## Step 1: Build & Push the Firewall Image

```bash
# Build
docker build -t your-registry.io/isartor:v0.1.0 -f docker/Dockerfile .

# Push
docker push your-registry.io/isartor:v0.1.0
```

---

## Step 2: Namespace & Secrets

```bash
kubectl create namespace isartor

# Cloud LLM API key (Layer 3 fallback)
kubectl create secret generic isartor-llm-secret \
  --namespace isartor \
  --from-literal=api-key='sk-...'

# Firewall API key (Layer 0 auth)
kubectl create secret generic isartor-gateway-secret \
  --namespace isartor \
  --from-literal=gateway-api-key='your-production-key'
```

---

## Step 3: Firewall Deployment

```yaml
# k8s/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isartor-gateway
  namespace: isartor
  labels:
    app: isartor-gateway
spec:
  replicas: 2
  selector:
    matchLabels:
      app: isartor-gateway
  template:
    metadata:
      labels:
        app: isartor-gateway
    spec:
      containers:
        - name: gateway
          image: your-registry.io/isartor:v0.1.0
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: ISARTOR__HOST_PORT
              value: "0.0.0.0:8080"
            - name: ISARTOR__GATEWAY_API_KEY
              valueFrom:
                secretKeyRef:
                  name: isartor-gateway-secret
                  key: gateway-api-key
            # Pluggable backends — scaled for multi-replica K8s
            - name: ISARTOR__CACHE_BACKEND
              value: "redis"          # Shared cache across all firewall pods
            - name: ISARTOR__REDIS_URL
              value: "redis://redis.isartor:6379"
            - name: ISARTOR__ROUTER_BACKEND
              value: "vllm"           # GPU-backed vLLM inference pool
            - name: ISARTOR__VLLM_URL
              value: "http://isartor-inference:8081"
            - name: ISARTOR__VLLM_MODEL
              value: "gemma-2-2b-it"
            # Cache
            - name: ISARTOR__CACHE_MODE
              value: "both"
            - name: ISARTOR__SIMILARITY_THRESHOLD
              value: "0.85"
            - name: ISARTOR__CACHE_TTL_SECS
              value: "300"
            - name: ISARTOR__CACHE_MAX_CAPACITY
              value: "50000"
            # Inference pool (internal service)
            - name: ISARTOR__LAYER2__SIDECAR_URL
              value: "http://isartor-inference:8081"
            - name: ISARTOR__LAYER2__MODEL_NAME
              value: "phi-3-mini"
            - name: ISARTOR__LAYER2__TIMEOUT_SECONDS
              value: "30"
            # Embedding pool (optional — default uses in-process candle)
            - name: ISARTOR__EMBEDDING_SIDECAR__SIDECAR_URL
              value: "http://isartor-embedding:8082"
            - name: ISARTOR__EMBEDDING_SIDECAR__MODEL_NAME
              value: "all-minilm"
            - name: ISARTOR__EMBEDDING_SIDECAR__TIMEOUT_SECONDS
              value: "10"
            # Layer 3 — Cloud LLM
            - name: ISARTOR__LLM_PROVIDER
              value: "openai"
            - name: ISARTOR__EXTERNAL_LLM_MODEL
              value: "gpt-4o-mini"
            - name: ISARTOR__EXTERNAL_LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: isartor-llm-secret
                  key: api-key
            # Observability
            - name: ISARTOR__ENABLE_MONITORING
              value: "true"
            - name: ISARTOR__OTEL_EXPORTER_ENDPOINT
              value: "http://otel-collector.isartor:4317"
          resources:
            requests:
              cpu: "250m"
              memory: "128Mi"
            limits:
              cpu: "1000m"
              memory: "256Mi"
          readinessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 10
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: isartor-gateway
  namespace: isartor
spec:
  selector:
    app: isartor-gateway
  ports:
    - port: 8080
      targetPort: http
      name: http
  type: ClusterIP
```

---

## Step 4: Inference Pool (vLLM)

[vLLM](https://github.com/vllm-project/vllm) provides high-throughput, GPU-optimised inference with continuous batching.

```yaml
# k8s/inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isartor-inference
  namespace: isartor
  labels:
    app: isartor-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: isartor-inference
  template:
    metadata:
      labels:
        app: isartor-inference
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "microsoft/Phi-3-mini-4k-instruct"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "8081"
            - "--max-model-len"
            - "4096"
            - "--gpu-memory-utilization"
            - "0.9"
          ports:
            - containerPort: 8081
              name: http
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: "8Gi"
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 10
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: isartor-inference
  namespace: isartor
spec:
  selector:
    app: isartor-inference
  ports:
    - port: 8081
      targetPort: http
      name: http
  type: ClusterIP
```

### Alternative: Text Generation Inference (TGI)

Replace vLLM with [TGI](https://github.com/huggingface/text-generation-inference) if you prefer Hugging Face's inference server:

```yaml
containers:
  - name: tgi
    image: ghcr.io/huggingface/text-generation-inference:latest
    args:
      - "--model-id"
      - "microsoft/Phi-3-mini-4k-instruct"
      - "--port"
      - "8081"
      - "--max-input-length"
      - "4096"
      - "--max-total-tokens"
      - "8192"
```

### Alternative: llama.cpp Server (CPU / Light GPU)

For budget clusters without heavy GPU nodes:

```yaml
containers:
  - name: llama-cpp
    image: ghcr.io/ggml-org/llama.cpp:server
    args:
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8081"
      - "--hf-repo"
      - "microsoft/Phi-3-mini-4k-instruct-gguf"
      - "--hf-file"
      - "Phi-3-mini-4k-instruct-q4.gguf"
      - "--ctx-size"
      - "4096"
      - "--n-gpu-layers"
      - "99"
```

---

## Step 5: Embedding Pool (TEI) — Optional

> **Note:** The gateway generates Layer 1 embeddings in-process via candle BertModel. This external embedding pool is optional for high-throughput deployments that want to offload embedding generation.

[Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) provides optimised embedding generation.

```yaml
# k8s/embedding-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isartor-embedding
  namespace: isartor
  labels:
    app: isartor-embedding
spec:
  replicas: 2
  selector:
    matchLabels:
      app: isartor-embedding
  template:
    metadata:
      labels:
        app: isartor-embedding
    spec:
      containers:
        - name: tei
          image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest
          args:
            - "--model-id"
            - "sentence-transformers/all-MiniLM-L6-v2"
            - "--port"
            - "8082"
          ports:
            - containerPort: 8082
              name: http
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "1Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: isartor-embedding
  namespace: isartor
spec:
  selector:
    app: isartor-embedding
  ports:
    - port: 8082
      targetPort: http
      name: http
  type: ClusterIP
```

---

## Step 6: Horizontal Pod Autoscaler

### Gateway HPA

```yaml
# k8s/gateway-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: isartor-gateway-hpa
  namespace: isartor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: isartor-gateway
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Pods
          value: 4
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 2
          periodSeconds: 120
```

### Inference Pool HPA (Custom Metrics)

For GPU-based scaling, use custom metrics from Prometheus:

```yaml
# k8s/inference-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: isartor-inference-hpa
  namespace: isartor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: isartor-inference
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "80"
```

> **Note:** GPU-based HPA requires the [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter) or KEDA to expose GPU metrics to the HPA controller.

---

## Step 7: Ingress

### nginx-ingress Example

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: isartor-ingress
  namespace: isartor
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.isartor.example.com
      secretName: isartor-tls
  rules:
    - host: api.isartor.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: isartor-gateway
                port:
                  number: 8080
```

### Istio VirtualService (Service Mesh)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: isartor-vs
  namespace: isartor
spec:
  hosts:
    - api.isartor.example.com
  gateways:
    - isartor-gateway
  http:
    - match:
        - uri:
            prefix: /api/
      route:
        - destination:
            host: isartor-gateway
            port:
              number: 8080
      timeout: 120s
      retries:
        attempts: 2
        perTryTimeout: 60s
```

---

## Step 8: Apply Everything

```bash
# Apply in order
kubectl apply -f k8s/gateway-deployment.yaml
kubectl apply -f k8s/inference-deployment.yaml
kubectl apply -f k8s/embedding-deployment.yaml
kubectl apply -f k8s/gateway-hpa.yaml
kubectl apply -f k8s/inference-hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Verify
kubectl get pods -n isartor
kubectl get svc -n isartor
kubectl get hpa -n isartor
```

---

## Redis Configuration for Distributed Cache

Enterprise deployments use Redis to share the exact-match cache across all firewall pods. Configure the cache provider via environment variables or `isartor.yaml`:

### Environment Variables

```bash
ISARTOR__CACHE_BACKEND=redis
ISARTOR__REDIS_URL=redis://redis-cluster.svc:6379
```

### YAML Configuration

```yaml
exact_cache:
  provider: redis
  redis_url: "redis://redis-cluster.svc:6379"
  # Optional: redis_db: 0
```

### Kubernetes Topology with Redis

Deploy Redis as a StatefulSet within the cluster, accessible only via ClusterIP:

```
[Ingress]
   |
[Isartor Deployment] <--> [Redis StatefulSet]
   |
   +--> [vLLM Deployment (GPU nodes)]
```

- Isartor pods scale horizontally for network I/O and cache hits.
- Redis ensures cache consistency across all pods.
- The vLLM GPU pool scales independently for inference throughput.

---

## vLLM Configuration for SLM Routing

Enterprise deployments replace the embedded candle SLM with a remote vLLM inference pool for higher throughput. Configure the router backend via environment variables or `isartor.yaml`:

### Environment Variables

```bash
ISARTOR__ROUTER_BACKEND=vllm
ISARTOR__VLLM_URL=http://vllm-openai.svc:8000
ISARTOR__VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
```

### YAML Configuration

```yaml
slm_router:
  provider: remote_http
  remote_url: "http://vllm-openai.svc:8000"
  model: "meta-llama/Llama-3-8B-Instruct"
```

### Docker Compose Example (Enterprise Sidecar)

For development or staging environments that mirror enterprise topology:

```yaml
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

---

## Observability in Level 3

For Kubernetes deployments, you have several options:

| Approach | Stack | Effort |
| --- | --- | --- |
| **Self-managed** | OTel Collector DaemonSet → Jaeger + Prometheus + Grafana | Medium |
| **Managed (AWS)** | AWS X-Ray + CloudWatch + Managed Grafana | Low |
| **Managed (GCP)** | Cloud Trace + Cloud Monitoring | Low |
| **Managed (Azure)** | Azure Monitor + Application Insights | Low |
| **Third-party** | Datadog / New Relic / Grafana Cloud | Low |

The gateway exports traces and metrics via OTLP gRPC to whatever `ISARTOR__OTEL_EXPORTER_ENDPOINT` points at. See [Metrics & Tracing](../observability/metrics-tracing.md) for detailed setup.

---

## Scalability Deep-Dive

Level 3 is designed for horizontal scaling. The Pluggable Trait Provider architecture ensures every component can scale independently:

### Stateless Gateway Pods

The Isartor gateway binary is **fully stateless** when configured with `cache_backend=redis` and `router_backend=vllm`. All request-scoped state (cache, inference) is offloaded to external services, meaning:

- **Gateway pods scale linearly** — add replicas via HPA without coordination overhead.
- **Zero warm-up penalty** — new pods serve requests immediately (no model loading, no cache priming).
- **Rolling updates** — deploy new versions with zero downtime; old and new pods share the same Redis cache.

### Shared Cache via Redis

With `ISARTOR__CACHE_BACKEND=redis`:

| Benefit | Impact |
| --- | --- |
| **Consistent hit rate** | All pods read/write the same cache — no per-pod cold caches |
| **Memory efficiency** | Cache memory is centralised, not duplicated N times |
| **Persistence** | Redis AOF/RDB survives pod restarts |
| **Cluster mode** | Redis Cluster or ElastiCache provides sharded, HA caching |

### GPU Inference Pool (vLLM)

With `ISARTOR__ROUTER_BACKEND=vllm`:

| Benefit | Impact |
| --- | --- |
| **Independent GPU scaling** | Scale inference replicas separately from gateway pods |
| **Continuous batching** | vLLM's PagedAttention maximises GPU utilisation |
| **Mixed hardware** | Gateway runs on cheap CPU nodes; inference on GPU nodes |
| **Cost control** | Scale inference to zero when idle (KEDA + queue-depth trigger) |

### Scaling Dimensions

| Dimension | Knob | Metric |
| --- | --- | --- |
| Gateway replicas | HPA `minReplicas` / `maxReplicas` | CPU utilisation, request rate |
| Inference replicas | HPA on custom GPU metrics | GPU utilisation, queue depth |
| Cache capacity | `ISARTOR__CACHE_MAX_CAPACITY` | Cache hit rate, memory usage |
| Concurrency | HPA + replica scaling | P95 latency, request rate |
| Redis | Redis Cluster nodes | Key count, memory, eviction rate |

---

## Cost Optimisation

| Strategy | Description |
| --- | --- |
| **Spot / preemptible nodes** | Use for inference pods (they're stateless and restart quickly) |
| **Scale-to-zero** | Use KEDA with queue-depth trigger to scale inference to 0 when idle |
| **Right-size GPU** | A100 80 GB for large models, T4/L4 for Phi-3-mini (4 GB VRAM is sufficient) |
| **Shared GPU** | NVIDIA MPS or MIG to run multiple inference pods per GPU |
| **Semantic cache** | Higher `ISARTOR__CACHE_MAX_CAPACITY` = fewer inference calls |
| **Smaller quantisation** | Q4_K_M uses less VRAM at marginal quality cost |

---

## Security Checklist

- [ ] TLS termination at ingress (cert-manager + Let's Encrypt or cloud certs)
- [ ] mTLS between services (Istio / Linkerd / Cilium)
- [ ] `ISARTOR__GATEWAY_API_KEY` from Kubernetes Secret, not plaintext
- [ ] `ISARTOR__EXTERNAL_LLM_API_KEY` from Kubernetes Secret
- [ ] Network policies restricting pod-to-pod communication
- [ ] RBAC: least-privilege ServiceAccounts for each workload
- [ ] Pod security standards: `restricted` or `baseline`
- [ ] Image scanning (Trivy, Snyk) in CI pipeline
- [ ] Audit logging enabled on the cluster

---

## Downgrading to Level 2

If Kubernetes overhead doesn't justify the scale:

1. Export your env vars from the Kubernetes ConfigMap/Secret.
2. Map them into `docker/.env.full`.
3. Run `docker compose -f docker-compose.sidecar.yml up --build`.

No code changes — the binary is identical across all three tiers.
