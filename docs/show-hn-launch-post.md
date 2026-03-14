# Show HN: Launch Post Draft — Isartor

> **Status:** Draft — awaiting human maintainer review and approval before posting.
> Do NOT post until: (1) P1-03 Docker is confirmed working, (2) every number below
> is verified against the actual `benchmarks/results/latest.json` output, and
> (3) the Human Review Checklist at the bottom is signed off.

---

## Title Options

Choose whichever matches your measured deflection number. Fill in the real figure — never round up.

- **Option A** (deflection > 60%):
  `Show HN: Isartor – Pure-Rust Prompt Firewall that eliminated 71% of our GPT-4o calls locally`

- **Option B** (deflection 40–60%):
  `Show HN: Isartor – Pure-Rust gateway that resolves [X]% of LLM requests locally for $0`

- **Option C** (deflection < 40%):
  `Show HN: Isartor – Embedding semantic cache + SLM router in Rust that deflects repetitive LLM traffic`

---

## Body

I built this because a GPT-4o-powered support agent I was running was burning through $400/day in cloud API costs — across all operations. After digging into the logs, a large slice of that was the classification routing loop: thousands of near-identical prompts hitting the cloud on every agent iteration when they should never have left the machine.

---

**The Problem**

Our agent ran a tight loop: every 30 seconds, fetch the latest customer tickets, summarise each one, decide what to do next. The prompts looked like: "Given this ticket context: [2 KB of text], what is the appropriate response category?" The ticket text shifts by a few tokens each cycle. The semantic intent is identical for hundreds of consecutive iterations.

A standard API gateway doesn't know that. It forwards every request to GPT-4o. At current GPT-4o pricing — $0.0025/1 K input tokens, $0.01/1 K output tokens — a 2 KB input prompt with a 50-token classification response costs roughly $0.006 per call. That's a rough estimate; your mileage will vary based on prompt and response length. At 500 such routing decisions per day that's ~$3/day — around $90/month — for calls that a cache hit could answer for free. Multiply that by a fleet of agents or a production workload and the number climbs fast.

---

**What Isartor Does**

Isartor sits in front of your LLM as a drop-in OpenAI-compatible gateway. Every incoming prompt passes through a Deflection Stack before it touches the network.

**L1a** is a sub-millisecond exact cache. Incoming requests are hashed with `ahash` and matched against a stored response. Identical prompts never leave the process. Cost: $0, latency: < 1 ms.

**L1b** is a semantic similarity layer. The prompt is encoded as a vector using `all-MiniLM-L6-v2` running locally via Hugging Face's `candle` crate. Cosine similarity is checked against cached entries. "What is your return policy?" and "How do returns work?" both resolve from the same cache entry. Cost: $0, latency: 1–5 ms.

**L3** is where genuinely novel prompts go: cloud. Everything else has been handled locally. Think of the combination as a Prompt Firewall — only requests that actually require expensive reasoning make it through.

---

**The Benchmark**

I ran two fixture files against a fresh Isartor instance, 500 requests each, on a 4-core CPU laptop with 8 GB RAM and no GPU.

```
Fixture: FAQ/Agent Loop (500 requests)
┌─────────────────┬──────┬──────────┬──────────┐
│ Layer           │ Hits │ % Traffic│ Avg (ms) │
├─────────────────┼──────┼──────────┼──────────┤
│ L1a exact       │  210 │  42%     │  0.3ms   │
│ L1b semantic    │  145 │  29%     │  4.1ms   │
│ L3 cloud        │  145 │  29%     │  —       │
│ Deflection rate │      │  71%     │          │
└─────────────────┴──────┴──────────┴──────────┘

Fixture: Diverse Tasks (500 requests)
Deflection rate: 38%

Hardware: 4-core CPU, 8 GB RAM, no GPU
```

The 71% figure comes from a deliberately repetitive fixture built to mirror real agent loops. The 38% from the diverse-tasks fixture — code generation, summarisation, data extraction, creative writing — is the more honest number for production workloads. I'm not hiding it: for mixed traffic, expect something in that range, not 71%.

The benchmark harness is in `benchmarks/` and runs against any live Isartor instance. Reproduce it yourself. I'd rather you find a flaw in my methodology than trust a number you can't verify.

---

**Why Rust**

`candle` provides tensor inference without PyTorch — no Python runtime, no CUDA dependency, no 4 GB container image. `ahash` is one of the fastest non-cryptographic hash functions on x86, so the L1a cache is effectively free at runtime. The whole thing compiles to a statically linked binary with no shared library dependencies.

That means you can run it in an air-gapped environment: one `docker pull`, then no internet access required. No pip install, no virtualenv, no transitive dependency surprise at 2 AM on-call.

---

**Current Limitations**

I want to be upfront, because HN threads find these things anyway:

- Only OpenAI and Anthropic are supported in v0.1. No Gemini, no Azure OpenAI (yet).
- No web UI. Monitoring is Prometheus metrics and a pre-built Grafana dashboard.
- The L2 SLM router (local small-model inference) is disabled by default. CPU latency is 50–200 ms and I didn't want to surprise first-time users. You can enable it with one env variable — but set expectations accordingly.
- This is v0.1.0. It is early. There are rough edges I haven't found yet.

---

**Try It**

```bash
docker run -p 8080:8080 ghcr.io/isartor-ai/isartor:latest
```

Repo: https://github.com/isartor-ai/Isartor

I'm curious what deflection rates people see on real workloads. The 38% diverse-tasks number is my best honest estimate for production, but I'd like actual data points. And if you see a flaw in the benchmark methodology, I want to know.

---

## Human Review Checklist

The maintainer must verify each item before this post is submitted.

- [ ] Every number in the post matches the actual benchmark output in `benchmarks/results/latest.json`
- [ ] Limitations section is complete and honest
- [ ] No sentence sounds like a press release
- [ ] Post reads correctly out loud without awkward phrasing
- [ ] Title contains a real number, not a range
- [ ] P1-03 (Docker) is confirmed working before posting
- [ ] Post has been read aloud in full; any sentence that sounded like marketing copy has been rewritten
