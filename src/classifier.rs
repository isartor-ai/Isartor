use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::ClassifierRoutingConfig;

fn default_artifact_version() -> u32 {
    1
}

pub fn normalize_label(label: &str) -> String {
    label.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassificationHeadArtifact {
    pub labels: Vec<String>,
    pub weights: Vec<Vec<f32>>,
    #[serde(default)]
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiHeadClassifierArtifact {
    #[serde(default = "default_artifact_version")]
    pub version: u32,
    pub embedding_dim: usize,
    pub task_type: ClassificationHeadArtifact,
    pub complexity: ClassificationHeadArtifact,
    pub persona: ClassificationHeadArtifact,
    pub domain: ClassificationHeadArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DimensionPrediction {
    pub label: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiHeadClassification {
    pub task_type: DimensionPrediction,
    pub complexity: DimensionPrediction,
    pub persona: DimensionPrediction,
    pub domain: DimensionPrediction,
    pub overall_confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierRouteDecision {
    pub selected_provider: Option<String>,
    pub selected_model: Option<String>,
    pub matched_rule_name: Option<String>,
    pub matched_rule_index: Option<usize>,
    pub classification: MultiHeadClassification,
}

impl ClassifierRouteDecision {
    pub fn cache_key_provider_fragment(&self) -> Option<&str> {
        self.selected_provider.as_deref()
    }
}

#[derive(Debug, Clone)]
pub struct MiniLmMultiHeadClassifier {
    artifact: MultiHeadClassifierArtifact,
}

impl MiniLmMultiHeadClassifier {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read classifier artifact {}", path.display()))?;
        let mut artifact: MultiHeadClassifierArtifact = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse classifier artifact {}", path.display()))?;
        validate_head("task_type", artifact.embedding_dim, &mut artifact.task_type)?;
        validate_head(
            "complexity",
            artifact.embedding_dim,
            &mut artifact.complexity,
        )?;
        validate_head("persona", artifact.embedding_dim, &mut artifact.persona)?;
        validate_head("domain", artifact.embedding_dim, &mut artifact.domain)?;
        Ok(Self { artifact })
    }

    pub fn classify_embedding(&self, embedding: &[f32]) -> Result<MultiHeadClassification> {
        if embedding.len() != self.artifact.embedding_dim {
            anyhow::bail!(
                "classifier embedding dimension mismatch: expected {}, got {}",
                self.artifact.embedding_dim,
                embedding.len()
            );
        }

        let task_type = predict_head(&self.artifact.task_type, embedding)?;
        let complexity = predict_head(&self.artifact.complexity, embedding)?;
        let persona = predict_head(&self.artifact.persona, embedding)?;
        let domain = predict_head(&self.artifact.domain, embedding)?;
        let overall_confidence =
            (task_type.confidence + complexity.confidence + persona.confidence + domain.confidence)
                / 4.0;

        Ok(MultiHeadClassification {
            task_type,
            complexity,
            persona,
            domain,
            overall_confidence,
        })
    }
}

pub fn select_route(
    config: &ClassifierRoutingConfig,
    classification: &MultiHeadClassification,
) -> Option<ClassifierRouteDecision> {
    if !config.enabled || classification.overall_confidence < config.confidence_threshold {
        return None;
    }

    let effective_rules = config.effective_rules();

    for (index, rule) in effective_rules.iter().enumerate() {
        let min_confidence = rule.min_confidence.unwrap_or(config.confidence_threshold);
        if classification.overall_confidence < min_confidence {
            continue;
        }
        if rule
            .provider
            .as_deref()
            .is_none_or(|provider| provider.trim().is_empty())
            && rule
                .model
                .as_deref()
                .is_none_or(|model| model.trim().is_empty())
        {
            continue;
        }
        if !rule_label_matches(rule.task_type.as_deref(), &classification.task_type.label)
            || !rule_label_matches(rule.complexity.as_deref(), &classification.complexity.label)
            || !rule_label_matches(rule.persona.as_deref(), &classification.persona.label)
            || !rule_label_matches(rule.domain.as_deref(), &classification.domain.label)
        {
            continue;
        }
        return Some(ClassifierRouteDecision {
            selected_provider: rule.provider.as_deref().map(normalize_label),
            selected_model: rule.model.clone().filter(|model| !model.trim().is_empty()),
            matched_rule_name: (!rule.name.trim().is_empty()).then(|| rule.name.clone()),
            matched_rule_index: Some(index),
            classification: classification.clone(),
        });
    }

    None
}

fn rule_label_matches(expected: Option<&str>, actual: &str) -> bool {
    expected
        .map(normalize_label)
        .is_none_or(|expected| expected == actual)
}

fn validate_head(
    head_name: &str,
    embedding_dim: usize,
    head: &mut ClassificationHeadArtifact,
) -> Result<()> {
    if embedding_dim == 0 {
        anyhow::bail!("classifier artifact embedding_dim must be greater than zero");
    }
    if head.labels.is_empty() {
        anyhow::bail!("classifier head {head_name} must define at least one label");
    }
    if head.weights.len() != head.labels.len() {
        anyhow::bail!(
            "classifier head {head_name} weight rows ({}) must match labels ({})",
            head.weights.len(),
            head.labels.len()
        );
    }
    for (index, row) in head.weights.iter().enumerate() {
        if row.len() != embedding_dim {
            anyhow::bail!(
                "classifier head {head_name} row {index} has dimension {}, expected {}",
                row.len(),
                embedding_dim
            );
        }
    }
    if head.bias.is_empty() {
        head.bias = vec![0.0; head.labels.len()];
    }
    if head.bias.len() != head.labels.len() {
        anyhow::bail!(
            "classifier head {head_name} bias entries ({}) must match labels ({})",
            head.bias.len(),
            head.labels.len()
        );
    }
    head.labels = head
        .labels
        .iter()
        .map(|label| normalize_label(label))
        .collect();
    Ok(())
}

fn predict_head(
    head: &ClassificationHeadArtifact,
    embedding: &[f32],
) -> Result<DimensionPrediction> {
    let mut logits = Vec::with_capacity(head.labels.len());
    for (index, weights) in head.weights.iter().enumerate() {
        let dot = weights
            .iter()
            .zip(embedding.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        logits.push(dot + head.bias[index]);
    }

    let probabilities = softmax(&logits);
    let (best_index, confidence) = probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .context("classifier head produced no scores")?;

    Ok(DimensionPrediction {
        label: head.labels[best_index].clone(),
        confidence,
    })
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |current, value| current.max(value));
    let exps = logits
        .iter()
        .map(|value| (value - max_logit).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|value| value / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture_classifier() -> MiniLmMultiHeadClassifier {
        MiniLmMultiHeadClassifier {
            artifact: MultiHeadClassifierArtifact {
                version: 1,
                embedding_dim: 3,
                task_type: ClassificationHeadArtifact {
                    labels: vec!["lookup".into(), "codegen".into()],
                    weights: vec![vec![2.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]],
                    bias: vec![0.0, 0.0],
                },
                complexity: ClassificationHeadArtifact {
                    labels: vec!["simple".into(), "complex".into()],
                    weights: vec![vec![1.5, 0.0, 0.0], vec![0.0, 1.5, 0.0]],
                    bias: vec![0.0, 0.0],
                },
                persona: ClassificationHeadArtifact {
                    labels: vec!["analyst".into(), "builder".into()],
                    weights: vec![vec![0.0, 0.0, 2.0], vec![0.0, 1.0, 0.0]],
                    bias: vec![0.0, 0.0],
                },
                domain: ClassificationHeadArtifact {
                    labels: vec!["docs".into(), "code".into()],
                    weights: vec![vec![0.0, 0.0, 1.5], vec![0.0, 1.5, 0.0]],
                    bias: vec![0.0, 0.0],
                },
            },
        }
    }

    #[test]
    fn classify_embedding_scores_all_heads() {
        let classification = fixture_classifier()
            .classify_embedding(&[0.1, 0.9, 0.2])
            .unwrap();
        assert_eq!(classification.task_type.label, "codegen");
        assert_eq!(classification.complexity.label, "complex");
        assert_eq!(classification.persona.label, "builder");
        assert_eq!(classification.domain.label, "code");
        assert!(classification.overall_confidence > 0.5);
    }

    #[test]
    fn select_route_matches_first_rule() {
        let classification = fixture_classifier()
            .classify_embedding(&[0.1, 0.9, 0.2])
            .unwrap();
        let config = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.5,
            fallback_to_existing_routing: true,
            rules: vec![
                crate::config::ClassifierRoutingRuleConfig {
                    name: "codegen-to-groq".into(),
                    task_type: Some("codegen".into()),
                    complexity: Some("complex".into()),
                    persona: None,
                    domain: Some("code".into()),
                    provider: Some("groq".into()),
                    model: Some("qwen/qwen3-32b".into()),
                    min_confidence: None,
                },
                crate::config::ClassifierRoutingRuleConfig {
                    name: "fallback".into(),
                    task_type: None,
                    complexity: None,
                    persona: None,
                    domain: None,
                    provider: Some("openai".into()),
                    model: Some("gpt-4o-mini".into()),
                    min_confidence: None,
                },
            ],
            matrix: HashMap::new(),
        };

        let route = select_route(&config, &classification).unwrap();
        assert_eq!(route.selected_provider.as_deref(), Some("groq"));
        assert_eq!(route.selected_model.as_deref(), Some("qwen/qwen3-32b"));
        assert_eq!(route.matched_rule_name.as_deref(), Some("codegen-to-groq"));
    }

    #[test]
    fn validate_head_requires_matching_dimensions() {
        let mut head = ClassificationHeadArtifact {
            labels: vec!["simple".into()],
            weights: vec![vec![1.0, 2.0]],
            bias: vec![0.0],
        };
        let err = validate_head("complexity", 3, &mut head).unwrap_err();
        assert!(err.to_string().contains("expected 3"));
    }

    #[test]
    fn normalize_label_lowercases_and_normalizes_spacing() {
        assert_eq!(normalize_label("Code Review"), "code_review");
        assert_eq!(normalize_label("Agent-Builder"), "agent_builder");
    }

    #[test]
    fn select_route_ignores_rules_without_provider_or_model_targets() {
        let classification = fixture_classifier()
            .classify_embedding(&[0.1, 0.9, 0.2])
            .unwrap();
        let config = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.5,
            fallback_to_existing_routing: true,
            rules: vec![crate::config::ClassifierRoutingRuleConfig {
                name: "labels-only".into(),
                task_type: Some("codegen".into()),
                complexity: Some("complex".into()),
                persona: Some("builder".into()),
                domain: Some("code".into()),
                provider: None,
                model: None,
                min_confidence: None,
            }],
            matrix: HashMap::new(),
        };

        assert!(select_route(&config, &classification).is_none());
    }

    #[test]
    fn matrix_compiles_to_rules_and_routes() {
        let classification = fixture_classifier()
            .classify_embedding(&[0.1, 0.9, 0.2])
            .unwrap();
        // classification: task_type=codegen, complexity=complex, persona=builder, domain=code

        let mut complex_map = HashMap::new();
        complex_map.insert("code_generation".into(), "groq/llama-3.3-70b".into());
        complex_map.insert("codegen".into(), "anthropic/claude-sonnet-4".into());
        let mut simple_map = HashMap::new();
        simple_map.insert("codegen".into(), "groq/llama-3.1-8b".into());

        let mut matrix = HashMap::new();
        matrix.insert("complex".into(), complex_map);
        matrix.insert("simple".into(), simple_map);

        let config = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.5,
            fallback_to_existing_routing: true,
            rules: vec![],
            matrix,
        };

        let route = select_route(&config, &classification).unwrap();
        assert_eq!(route.selected_provider.as_deref(), Some("anthropic"));
        assert_eq!(route.selected_model.as_deref(), Some("claude-sonnet-4"));
        assert!(
            route
                .matched_rule_name
                .as_deref()
                .unwrap()
                .starts_with("matrix:")
        );
    }

    #[test]
    fn matrix_local_entry_is_skipped() {
        let classification = fixture_classifier()
            .classify_embedding(&[0.1, 0.9, 0.2])
            .unwrap();

        let mut complex_map = HashMap::new();
        complex_map.insert("codegen".into(), "local".into());

        let mut matrix = HashMap::new();
        matrix.insert("complex".into(), complex_map);

        let config = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.5,
            fallback_to_existing_routing: true,
            rules: vec![],
            matrix,
        };

        // "local" entries are skipped, so no route should match
        assert!(select_route(&config, &classification).is_none());
    }

    #[test]
    fn matrix_default_key_acts_as_wildcard() {
        let classification = fixture_classifier()
            .classify_embedding(&[0.1, 0.9, 0.2])
            .unwrap();
        // classification: task_type=codegen, complexity=complex

        let mut default_map = HashMap::new();
        default_map.insert("default".into(), "openai/gpt-4o-mini".into());

        let mut matrix = HashMap::new();
        matrix.insert("default".into(), default_map);

        let config = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.5,
            fallback_to_existing_routing: true,
            rules: vec![],
            matrix,
        };

        let route = select_route(&config, &classification).unwrap();
        assert_eq!(route.selected_provider.as_deref(), Some("openai"));
        assert_eq!(route.selected_model.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn explicit_rules_take_priority_over_matrix() {
        let classification = fixture_classifier()
            .classify_embedding(&[0.1, 0.9, 0.2])
            .unwrap();

        let mut complex_map = HashMap::new();
        complex_map.insert("codegen".into(), "anthropic/claude-sonnet-4".into());

        let mut matrix = HashMap::new();
        matrix.insert("complex".into(), complex_map);

        let config = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.5,
            fallback_to_existing_routing: true,
            rules: vec![crate::config::ClassifierRoutingRuleConfig {
                name: "explicit-groq".into(),
                task_type: Some("codegen".into()),
                complexity: Some("complex".into()),
                persona: None,
                domain: None,
                provider: Some("groq".into()),
                model: Some("llama-3.3-70b".into()),
                min_confidence: None,
            }],
            matrix,
        };

        let route = select_route(&config, &classification).unwrap();
        // Explicit rule wins over matrix
        assert_eq!(route.selected_provider.as_deref(), Some("groq"));
        assert_eq!(route.matched_rule_name.as_deref(), Some("explicit-groq"));
    }

    #[test]
    fn effective_rules_orders_specific_before_default() {
        let mut complex_map = HashMap::new();
        complex_map.insert("codegen".into(), "groq/llama-3.3-70b".into());
        complex_map.insert("default".into(), "openai/gpt-4o".into());
        let mut default_map = HashMap::new();
        default_map.insert("default".into(), "openai/gpt-4o-mini".into());

        let mut matrix = HashMap::new();
        matrix.insert("complex".into(), complex_map);
        matrix.insert("default".into(), default_map);

        let config = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.5,
            fallback_to_existing_routing: true,
            rules: vec![],
            matrix,
        };

        let rules = config.effective_rules();
        // Specific (complex/codegen) comes before partial-default (complex/default)
        // and before catch-all (default/default)
        let names: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
        let specific_pos = names
            .iter()
            .position(|n| *n == "matrix:complex/codegen")
            .unwrap();
        let partial_pos = names
            .iter()
            .position(|n| *n == "matrix:complex/default")
            .unwrap();
        let catchall_pos = names
            .iter()
            .position(|n| *n == "matrix:default/default")
            .unwrap();
        assert!(specific_pos < partial_pos);
        assert!(partial_pos < catchall_pos);
    }
}
