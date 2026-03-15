# Why Most LLM Gateways Can't Pass a FedRAMP Review (And What Can)

*Published on the Isartor blog — targeting platform engineers and security architects at regulated enterprises.*

---

## The CISO's Nightmare

Picture this: a CISO at a federal agency is six months into an LLM gateway
evaluation. The vendor has given assurances — "our gateway is secure, all
data stays in your environment." The compliance team runs a network capture
during the proof-of-concept. Three unexpected domains light up:

- `telemetry.vendor.io` — anonymous usage metrics
- `license.vendor.io` — license key validation on every startup
- `registry.vendor.io` — model version checks

The FedRAMP audit fails. The project is cancelled. Six months of engineering
work discarded because nobody read the gateway's egress behavior carefully
enough before the evaluation began.

This is not a hypothetical. It happens routinely in regulated environments.
The mistake is usually honest — gateway teams build their products for
cloud-native deployments and add telemetry and license checks as an
afterthought, without thinking about what happens when those systems need
to run in an air-gapped facility.

---

## The Hidden Phone-Home Problem

Most LLM gateways have outbound connection patterns that are not documented
in their README. Let's be specific about what these are and why each one is a
blocker in a FedRAMP or HIPAA environment:

**License validation servers.** A gateway that validates its licence key
against a remote server cannot operate in a network segment with no outbound
internet access. Worse, the validation traffic typically contains the licence
key and the server's hostname — both of which may be considered sensitive data
in a classified environment. Under FedRAMP Moderate, SC-7 (Boundary
Protection) requires that external connections be explicitly authorised and
documented. An undocumented licence-check endpoint fails this control.

**Anonymous usage telemetry.** Many open-source gateways ship with
opt-out telemetry that sends aggregate usage statistics to the developer's
servers. Even "anonymous" telemetry can include prompt length distributions,
model names, or error rates that a regulated environment may consider
sensitive. Under HIPAA, any data that could be used to identify a patient
— including metadata about the prompts that process PHI — must stay within
the covered entity's environment.

**Model registry lookups.** Gateways that support automatic model updates
or capability discovery make outbound calls to check for new model versions.
In an air-gapped environment, there is no path for these calls to succeed —
and if the gateway blocks on a registry timeout, latency spikes cascade
through the application.

**OTel exporters enabled by default.** OpenTelemetry is essential for
observability, but a gateway that ships with `OTLP_EXPORTER_ENDPOINT` pointing
at a cloud-hosted collector creates a data exfiltration risk. Trace data
contains prompt content, response content, latency, and error messages. An
OTel exporter sending this to an external endpoint in a HIPAA environment
would be a reportable breach.

Each of these problems has the same root cause: the gateway was designed for
cloud-native deployments and retrofitted for security requirements, rather than
designed with air-gap constraints from the start.

---

## What "Truly Air-Gapped" Actually Means

A gateway that can genuinely pass an air-gap review must satisfy three
requirements:

**1. A static binary with no runtime dependencies.**
Every runtime dependency — a Python interpreter, a Node.js runtime, a JVM —
is a potential attack surface and a source of unexpected network calls. A
statically compiled binary eliminates the entire class of "your dependency
phoned home without you knowing" vulnerabilities. It also eliminates the
download-on-first-run pattern where models or plugins are fetched from the
internet when the gateway starts.

**2. Offline licence validation.**
Licence validation must work without a network call. The correct approach is
HMAC-based offline validation: the licence key embeds a cryptographic
signature that the binary verifies locally using a public key baked in at
compile time. No server call required. No licence-check traffic to document
in your FedRAMP boundary diagram.

**3. All models bundled — no download on first run.**
Any model that is downloaded at runtime creates a bootstrap dependency on
internet connectivity. For an air-gapped deployment, all models must be
available in the container image (or on a mounted volume) before the gateway
starts. This is non-negotiable for environments where the deployment system
has no outbound internet access at all.

Isartor is designed to meet all three requirements. The binary is compiled with Rust's
`--target x86_64-unknown-linux-musl` producing a fully static binary with
zero shared library dependencies. Licence validation uses HMAC offline
verification. The `latest-airgapped` Docker image is built to pre-bundle (or
pre-cache) all embedding models so that, once the image is transferred to the
air-gapped environment and `ISARTOR__OFFLINE_MODE=true` is set, no additional
model downloads or outbound internet access are required at runtime.
---

## The Configuration

Here is the complete environment variable configuration for a compliant
air-gapped deployment of Isartor in front of a self-hosted vLLM instance:

```bash
# ── Air-gap enforcement ──────────────────────────────────────────────
# Block all outbound cloud connections at the application layer.
export ISARTOR__OFFLINE_MODE=true

# ── Internal LLM routing (L3) ────────────────────────────────────────
# Route surviving cache-misses to your internal model server.
export ISARTOR__EXTERNAL_LLM_URL=http://vllm.internal.corp:8000/v1
export ISARTOR__LLM_PROVIDER=openai          # vLLM exposes OpenAI-compat API
export ISARTOR__EXTERNAL_LLM_MODEL=meta-llama/Llama-3-8B-Instruct

# ── Observability (internal collector only) ──────────────────────────
export ISARTOR__ENABLE_MONITORING=true
export ISARTOR__OTEL_EXPORTER_ENDPOINT=http://otel-collector.internal.corp:4317
```

Running `isartor connectivity-check` with this configuration produces:

```
Isartor Connectivity Audit
──────────────────────────
Required (L3 cloud routing):
  → http://vllm.internal.corp:8000/v1  [CONFIGURED]
    (BLOCKED — offline mode active)

Optional (observability / monitoring):
  → http://otel-collector.internal.corp:4317  [CONFIGURED]

Internal only (no external):
  → (in-memory cache — no network connection)  [CONFIGURED - internal]

Zero hidden telemetry connections: ✓ VERIFIED
Air-gap compatible: ✓ YES (L3 disabled or offline mode active)
```

This output is the screenshot your compliance team needs. Every connection
Isartor makes is explicit, documented, and internal.

---

## The FedRAMP Control Mapping

Understanding how a deployment posture maps to specific NIST 800-53 controls
is what separates a security claim from a security argument. Here are the four
controls most directly supported by Isartor's air-gapped deployment posture:

**AU-2 (Audit Logging):** AU-2 requires that the system generate audit records
for events relevant to security. Isartor logs every prompt, every deflection
decision, and every L3 forwarding event as a structured JSON record with a
distributed tracing span. The logs include the layer that handled the request
(L1a, L1b, L2, L3), the latency, and whether the request was deflected or
forwarded. These records can be ingested by any SIEM that accepts JSON log
streams.

**SC-7 (Boundary Protection):** SC-7 requires the system to monitor and
control communications at external boundary points. `ISARTOR__OFFLINE_MODE=true`
implements a hard application-layer block on all outbound connections to
non-internal endpoints. This is verified by the phone-home audit test in
`tests/phone_home_audit.rs`, which runs on every commit to `main` in CI. The
CI badge on the repository proves continuous enforcement.

**SI-4 (Information System Monitoring):** SI-4 requires monitoring of the
information system to detect attacks and indicators of compromise.
Isartor's OpenTelemetry integration exports traces and metrics to an internal
collector. The deflection stack metrics — cache hit rate, L3 call rate,
latency per layer — provide a real-time signal that can be baselined and
alerted on. An anomalous spike in L3 calls could indicate a cache poisoning
attempt.

**CM-6 (Configuration Settings):** CM-6 requires the organisation to establish
and document configuration settings. Every Isartor configuration parameter is
controlled by an environment variable with a documented default and a
documented security implication. The `ISARTOR__OFFLINE_MODE` flag, in
particular, has a documented effect: it is a single switch that moves the
system from "possibly communicates with cloud" to "provably does not
communicate with cloud."

---

## Call to Action

If you are a platform engineer or security architect at a regulated enterprise
evaluating LLM gateway options, start here:

1. Read the [Air-Gapped Deployment Guide](https://github.com/isartor-ai/Isartor/blob/main/docs/AIR-GAPPED.md)
   for the complete pre-deployment checklist.
2. Pull `ghcr.io/isartor-ai/isartor:latest-airgapped` and run
   `isartor connectivity-check` in your environment.
3. Review the [phone-home audit test](https://github.com/isartor-ai/Isartor/blob/main/tests/phone_home_audit.rs)
   to understand exactly what is being verified in CI.
4. Open an issue on [GitHub](https://github.com/isartor-ai/Isartor/issues)
   if you have compliance requirements not covered here — FedRAMP High, IL5,
   ITAR, and sector-specific requirements are all on the roadmap.

The binary that passes your network capture is the binary that passes your
FedRAMP review.
