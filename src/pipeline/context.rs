// =============================================================================
// Pipeline Context — The "State" flowing through the Deflection Stack.
//
// This is the central request context object that passes from layer to layer
// in the unidirectional Deflection Stack. Each algorithmic stage enriches
// the context with its computed outputs.
// =============================================================================

use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ── Intent Classification ────────────────────────────────────────────

/// The result of Layer 2's intent classification.
///
/// Determines how the pipeline routes the request: simple tasks are
/// handled locally by the SLM, complex tasks proceed through context
/// optimisation (Layer 2.5) and ultimately to the external LLM (Layer 3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentClassification {
    /// Not yet classified — initial state before Layer 2 runs.
    Unknown,
    /// Task is simple enough for the local SLM to handle directly.
    Simple,
    /// Task requires deep reasoning, code generation, or multi-step analysis.
    Complex,
    /// Task involves retrieval-augmented generation with external knowledge.
    Rag,
    /// Task involves code generation or analysis.
    CodeGen,
    /// Classification failed; treat as complex for safety.
    Unclassifiable,
}

impl fmt::Display for IntentClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => write!(f, "UNKNOWN"),
            Self::Simple => write!(f, "SIMPLE"),
            Self::Complex => write!(f, "COMPLEX"),
            Self::Rag => write!(f, "RAG"),
            Self::CodeGen => write!(f, "CODEGEN"),
            Self::Unclassifiable => write!(f, "UNCLASSIFIABLE"),
        }
    }
}

// ── Processing Log ───────────────────────────────────────────────────

/// A single observability entry in the pipeline's processing log.
///
/// Records what happened at each layer, how long it took, and whether
/// it short-circuited the pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessingLogEntry {
    /// Human-readable name of the pipeline stage (e.g. "Layer1_SemanticCache").
    pub stage: String,

    /// What action was taken at this stage.
    pub action: String,

    /// Wall-clock duration of this stage in milliseconds.
    pub duration_ms: u64,

    /// Whether this stage produced the final response (short-circuited).
    pub short_circuited: bool,

    /// Optional metadata (e.g. similarity score, intent label).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// ── Pipeline Response ────────────────────────────────────────────────

/// The final response produced by whichever pipeline layer resolved the request.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineResponse {
    /// Which layer produced this response (0, 1, 2, 3).
    pub resolved_by_layer: u8,

    /// The actual response text.
    pub message: String,

    /// Model name used to generate the response, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Total pipeline execution time in milliseconds.
    pub total_duration_ms: u64,

    /// Full processing log for observability.
    pub processing_log: Vec<ProcessingLogEntry>,
}

// ── Pipeline Context ─────────────────────────────────────────────────

/// Central request context that flows through every layer of the pipeline.
///
/// Each algorithmic stage reads from and writes to this context. The
/// orchestrator passes a mutable reference through the pipeline so that
/// downstream stages can see what upstream stages computed.
///
/// ```text
///  ┌──────────────────────────────────────────────────────────────┐
///  │  PipelineContext                                             │
///  │                                                             │
///  │  original_prompt ──► [Layer 1: Embedder] ──► request_vector │
///  │  request_vector  ──► [Layer 1: VectorStore] ──► cache hit?  │
///  │  original_prompt ──► [Layer 2: Classifier]  ──► intent      │
///  │  intent          ──► [Layer 2: Executor]    ──► response?   │
///  │  documents       ──► [Layer 2.5: Reranker]  ──► top-K docs  │
///  │  prompt + docs   ──► [Layer 3: LLM]         ──► response    │
///  └──────────────────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone)]
pub struct PipelineContext {
    // ── Immutable input ─────────────────────────────────────────
    /// The raw prompt string as received from the client.
    pub original_prompt: String,

    /// Unique request identifier for tracing / correlation.
    pub request_id: String,

    // ── Layer 1 outputs (Semantic Cache) ────────────────────────
    /// Dense embedding vector computed by the `Embedder` at Layer 1.
    /// `None` until Layer 1 runs.
    pub request_vector: Option<Vec<f64>>,

    // ── Layer 2 outputs (SLM Router) ────────────────────────────
    /// Intent classification result from the `IntentClassifier`.
    pub intent_classification: IntentClassification,

    /// Confidence score from the classifier (0.0–1.0).
    /// Higher means the classifier is more confident in its label.
    pub complexity_score: f64,

    // ── Layer 2.5 outputs (Context Optimiser) ───────────────────
    /// Raw candidate documents retrieved from the knowledge base
    /// *before* reranking. Populated just prior to Layer 2.5.
    pub retrieved_context_documents: Vec<String>,

    /// The optimised (reranked) subset of documents that will be
    /// included in the final LLM prompt. Populated by Layer 2.5.
    pub optimised_context_documents: Vec<String>,

    // ── Observability ───────────────────────────────────────────
    /// Ordered log of every processing step for full auditability.
    pub processing_log: Vec<ProcessingLogEntry>,

    /// Timestamp when the pipeline context was created.
    pub created_at: Instant,
}

impl PipelineContext {
    /// Create a new pipeline context for an incoming request.
    pub fn new(prompt: String) -> Self {
        Self {
            original_prompt: prompt,
            request_id: uuid_v4(),
            request_vector: None,
            intent_classification: IntentClassification::Unknown,
            complexity_score: 0.0,
            retrieved_context_documents: Vec::new(),
            optimised_context_documents: Vec::new(),
            processing_log: Vec::new(),
            created_at: Instant::now(),
        }
    }

    /// Append an entry to the processing log.
    pub fn log_step(
        &mut self,
        stage: impl Into<String>,
        action: impl Into<String>,
        duration_ms: u64,
        short_circuited: bool,
        metadata: Option<serde_json::Value>,
    ) {
        self.processing_log.push(ProcessingLogEntry {
            stage: stage.into(),
            action: action.into(),
            duration_ms,
            short_circuited,
            metadata,
        });
    }

    /// Total elapsed time since the context was created.
    pub fn elapsed_ms(&self) -> u64 {
        self.created_at.elapsed().as_millis() as u64
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Minimal UUID v4 generator (no external crate needed).
/// Produces a hex string suitable for request correlation.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    // Mix in a thread-local counter for uniqueness within the same nanosecond.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    format!("{:016x}-{:04x}", nanos, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── IntentClassification Display ─────────────────────────────

    #[test]
    fn intent_classification_display() {
        assert_eq!(IntentClassification::Unknown.to_string(), "UNKNOWN");
        assert_eq!(IntentClassification::Simple.to_string(), "SIMPLE");
        assert_eq!(IntentClassification::Complex.to_string(), "COMPLEX");
        assert_eq!(IntentClassification::Rag.to_string(), "RAG");
        assert_eq!(IntentClassification::CodeGen.to_string(), "CODEGEN");
        assert_eq!(
            IntentClassification::Unclassifiable.to_string(),
            "UNCLASSIFIABLE"
        );
    }

    #[test]
    fn intent_classification_eq() {
        assert_eq!(IntentClassification::Simple, IntentClassification::Simple);
        assert_ne!(IntentClassification::Simple, IntentClassification::Complex);
    }

    #[test]
    fn intent_classification_clone() {
        let ic = IntentClassification::Rag;
        let cloned = ic.clone();
        assert_eq!(ic, cloned);
    }

    #[test]
    fn intent_classification_serde_roundtrip() {
        let ic = IntentClassification::CodeGen;
        let json = serde_json::to_string(&ic).unwrap();
        let back: IntentClassification = serde_json::from_str(&json).unwrap();
        assert_eq!(back, IntentClassification::CodeGen);
    }

    // ── PipelineContext ──────────────────────────────────────────

    #[test]
    fn pipeline_context_new() {
        let ctx = PipelineContext::new("Hello world".into());
        assert_eq!(ctx.original_prompt, "Hello world");
        assert!(!ctx.request_id.is_empty());
        assert!(ctx.request_vector.is_none());
        assert_eq!(ctx.intent_classification, IntentClassification::Unknown);
        assert_eq!(ctx.complexity_score, 0.0);
        assert!(ctx.retrieved_context_documents.is_empty());
        assert!(ctx.optimised_context_documents.is_empty());
        assert!(ctx.processing_log.is_empty());
    }

    #[test]
    fn pipeline_context_unique_ids() {
        let c1 = PipelineContext::new("a".into());
        let c2 = PipelineContext::new("b".into());
        assert_ne!(c1.request_id, c2.request_id);
    }

    #[test]
    fn pipeline_context_log_step() {
        let mut ctx = PipelineContext::new("test".into());
        ctx.log_step("Layer1", "cache_miss", 5, false, None);
        ctx.log_step(
            "Layer2",
            "classified_simple",
            10,
            true,
            Some(serde_json::json!({"confidence": 0.95})),
        );

        assert_eq!(ctx.processing_log.len(), 2);
        assert_eq!(ctx.processing_log[0].stage, "Layer1");
        assert_eq!(ctx.processing_log[0].action, "cache_miss");
        assert_eq!(ctx.processing_log[0].duration_ms, 5);
        assert!(!ctx.processing_log[0].short_circuited);
        assert!(ctx.processing_log[0].metadata.is_none());

        assert_eq!(ctx.processing_log[1].stage, "Layer2");
        assert!(ctx.processing_log[1].short_circuited);
        assert!(ctx.processing_log[1].metadata.is_some());
    }

    #[test]
    fn pipeline_context_elapsed_ms() {
        let ctx = PipelineContext::new("test".into());
        // Just ensure it doesn't panic and returns >= 0.
        let _ = ctx.elapsed_ms();
    }

    // ── ProcessingLogEntry serialization ─────────────────────────

    #[test]
    fn processing_log_entry_serialize_no_metadata() {
        let entry = ProcessingLogEntry {
            stage: "Layer1".into(),
            action: "embed".into(),
            duration_ms: 42,
            short_circuited: false,
            metadata: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(!json.contains("metadata"));
    }

    #[test]
    fn processing_log_entry_serialize_with_metadata() {
        let entry = ProcessingLogEntry {
            stage: "Layer2".into(),
            action: "classify".into(),
            duration_ms: 100,
            short_circuited: false,
            metadata: Some(serde_json::json!({"intent": "SIMPLE"})),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"metadata\""));
        assert!(json.contains("SIMPLE"));
    }

    // ── PipelineResponse serialization ───────────────────────────

    #[test]
    fn pipeline_response_serialize() {
        let resp = PipelineResponse {
            resolved_by_layer: 2,
            message: "Hello".into(),
            model: Some("phi-3".into()),
            total_duration_ms: 150,
            processing_log: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"resolved_by_layer\":2"));
        assert!(json.contains("\"model\":\"phi-3\""));
    }

    #[test]
    fn pipeline_response_serialize_no_model() {
        let resp = PipelineResponse {
            resolved_by_layer: 1,
            message: "Cached".into(),
            model: None,
            total_duration_ms: 5,
            processing_log: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("\"model\""));
    }

    // ── uuid_v4 helper ───────────────────────────────────────────

    #[test]
    fn uuid_v4_generates_unique_ids() {
        let ids: Vec<String> = (0..100).map(|_| uuid_v4()).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), 100, "uuid_v4 should produce unique IDs");
    }
}
