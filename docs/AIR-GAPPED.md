# Air-Gapped Deployment Guide

## Overview

Isartor is architecturally the most air-gap-friendly LLM gateway available.
Its pure-Rust statically compiled binary embeds all inference models at build
time, requires no runtime dependencies, and validates licenses with an offline
HMAC check — so Isartor itself does not initiate unsolicited telemetry or
license calls to external services.

The zero-phone-home guarantee applies to Isartor-managed network paths: the
`--offline` flag disables L3 cloud routing and external observability backends
at the application layer, and our CI phone-home audit test (see
`tests/phone_home_audit.rs`) exercises these code paths on every commit.

Supported regulated industries: **defense**, **healthcare (HIPAA)**,
**finance (SOX)**, and **government (FedRAMP)**.

---

## Pre-Deployment Checklist

Complete these steps before deploying Isartor in an air-gapped environment:

1. **Download the airgapped Docker image**

   ```bash
   docker pull ghcr.io/isartor-ai/isartor:latest-airgapped
   ```

   This image includes local copies of the L1b embedding models to minimize
   or avoid external downloads during normal operation in most setups.
   See [Image Size Comparison](#image-size-comparison) for size details and
   be sure to follow any additional configuration steps required by your
   environment to operate fully offline.

2. **Transfer to your air-gapped environment** via your organisation's
   approved media transfer process (USB, air-gap data diode, etc.).

3. **Enable offline mode**

   ```bash
   export ISARTOR__OFFLINE_MODE=true
   ```

   Alternatively, pass `--offline` on the command line:

   ```bash
   isartor --offline
   ```

4. **Disable L3 or point it at an internal LLM endpoint**

   - For strictly air-gapped / zero-egress deployments, you **must** enable offline mode (step 3). Leaving `ISARTOR__EXTERNAL_LLM_API_KEY` unset *alone* does **not** prevent the gateway from attempting outbound L3 calls to the default external endpoint on cache misses.
   - To run fully local (cache + SLM only) with no outbound attempts, enable offline mode **and** leave `ISARTOR__EXTERNAL_LLM_API_KEY` unset.
   - To route L3 to a self-hosted model, see [Connecting to an Internal LLM](#connecting-to-an-internal-llm).

5. **Run `isartor connectivity-check`** to confirm zero external connections:

   ```bash
   isartor connectivity-check
   ```

   Expected output (with offline mode active):

   ```
   Isartor Connectivity Audit
   ──────────────────────────
   Required (L3 cloud routing):
     → api.openai.com:443     [NOT CONFIGURED]
       (BLOCKED — offline mode active)

   Optional (observability / monitoring):
     → http://localhost:4317  [NOT CONFIGURED]

   Internal only (no external):
     → (in-memory cache — no network connection)  [CONFIGURED - internal]

   Zero hidden telemetry connections: ✓ VERIFIED
   Air-gap compatible: ✓ YES (L3 disabled or offline mode active)
   ```

6. **Run `isartor audit verify`** *(planned — see issue #3)* to confirm the
   signed audit log is functioning correctly.

---

## Connecting to an Internal LLM

In this configuration Isartor acts as a fully air-gapped deflection layer in
front of an internal LLM. 100% of traffic stays inside the perimeter: L1a
and L1b handle cached / semantically similar prompts locally, and only genuine
cache misses are forwarded to your self-hosted model over the internal network.

```bash
# Route L3 to a self-hosted vLLM instance on the internal network.
export ISARTOR__EXTERNAL_LLM_URL=http://vllm.internal.corp:8000/v1
export ISARTOR__LLM_PROVIDER=openai          # vLLM exposes an OpenAI-compat API
export ISARTOR__EXTERNAL_LLM_MODEL=meta-llama/Llama-3-8B-Instruct

# Enable offline mode to block any accidental external connections.
export ISARTOR__OFFLINE_MODE=true

# Start the gateway.
isartor
```

> **Note:** `ISARTOR__EXTERNAL_LLM_URL` sets the L3 endpoint URL. Point it
> at your internal vLLM or TGI server.

With this configuration:

- L1a (exact cache) deflects duplicate prompts instantly (< 1 ms).
- L1b (semantic cache) deflects semantically similar prompts (1–5 ms).
- L3 forwards surviving cache-miss prompts to your internal vLLM.
- Zero bytes leave the network perimeter.

---

## Startup Status Banner

When offline mode is active, Isartor prints a status banner at startup so
operators can confirm the configuration at a glance:

```
  ┌──────────────────────────────────────────────────────┐
  │  [Isartor] OFFLINE MODE ACTIVE                       │
  ├──────────────────────────────────────────────────────┤
  │  ✓ L1a Exact Cache:     active                       │
  │  ✓ L1b Semantic Cache:  active                       │
  │  - L2 SLM Router:       disabled (ENABLE_SLM_ROUTER=false)│
  │  ✗ L3 Cloud Logic:      DISABLED (offline mode)      │
  │  ✗ Telemetry export:    DISABLED if external endpoint │
  │  ✓ License validation:  offline HMAC check           │
  └──────────────────────────────────────────────────────┘
```

---

## Environment Variables Reference

| Variable | Default | Description |
|:---------|:--------|:------------|
| `ISARTOR__OFFLINE_MODE` | `false` | Enable air-gap mode. Blocks L3 cloud calls. |
| `ISARTOR__EXTERNAL_LLM_URL` | — | Internal LLM endpoint (vLLM, TGI, etc.). |
| `ISARTOR__EXTERNAL_LLM_MODEL` | `gpt-4o-mini` | Model name passed to the internal LLM. |
| `ISARTOR__SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold for L1b cache hits. Lower values increase local deflection. |
| `ISARTOR__OTEL_EXPORTER_ENDPOINT` | `http://localhost:4317` | OTel collector endpoint. External URLs are suppressed in offline mode. |

---

## Image Size Comparison

| Image | Tag | Includes models | Compressed size |
|:------|:----|:----------------|:----------------|
| Base | `latest` | No (downloads on first run) | ~120 MB |
| Air-gapped | `latest-airgapped` | Yes (all-MiniLM-L6-v2 embedded) | ~210 MB |

The `latest-airgapped` image is approximately 90 MB larger due to the
pre-bundled embedding model. This is the recommended image for any environment
with restricted outbound internet access.

---

## Compliance Notes

### FedRAMP / NIST 800-53

This deployment posture supports the following NIST 800-53 controls:

| Control | Description | How Isartor Supports It |
|:--------|:------------|:------------------------|
| **AU-2** | Audit Logging | Every prompt, deflection decision, and L3 call is logged as a structured JSON event with tracing spans. |
| **SC-7** | Boundary Protection | `ISARTOR__OFFLINE_MODE=true` enforces a hard block on all outbound connections. The phone-home audit CI test verifies this. |
| **SI-4** | Information System Monitoring | OpenTelemetry traces + Prometheus metrics provide real-time visibility into the deflection stack. Internal-only OTel endpoints are supported. |
| **CM-6** | Configuration Settings | All settings are controlled via environment variables with documented defaults. No runtime code changes are needed. |

### HIPAA

When `ISARTOR__OFFLINE_MODE=true` and L3 is pointed at an internal model:

- PHI in prompts **never leaves the network perimeter**.
- The L1b semantic cache computes embeddings in-process using a pure-Rust
  `candle` model — no external API calls.
- Audit logs are written to stdout for ingestion by your internal SIEM.

### Disclaimer

This document describes deployment architecture. The controls described above
are architectural claims based on code behaviour — they are not a formal
compliance certification. Consult your compliance team and engage a qualified
assessor for formal FedRAMP authorization or HIPAA compliance review.

---

## Further Reading

- [Architecture overview](2-ARCHITECTURE.md)
- [Enterprise deployment guide](3-ENTERPRISE-GUIDE.md)
- [Configuration reference](5-CONFIGURATION-REF.md)
- [Phone-home audit test source](../tests/phone_home_audit.rs)
