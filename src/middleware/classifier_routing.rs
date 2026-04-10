use std::sync::Arc;

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use bytes::Bytes;

use crate::classifier::{ClassifierRouteDecision, MultiHeadClassification, select_route};
use crate::core::prompt::{extract_classifier_context, override_request_model};
use crate::middleware::body_buffer::BufferedBody;
use crate::state::AppState;

/// Runs the optional MiniLM multi-head classifier before cache lookup so routed
/// model/provider choices can influence downstream cache keys and Layer 3 routing.
pub async fn classifier_routing_middleware(mut request: Request, next: Next) -> Response {
    let state = match request.extensions().get::<Arc<AppState>>() {
        Some(state) => state.clone(),
        None => {
            tracing::error!("Classifier routing: AppState missing from request extensions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Firewall misconfiguration: missing application state",
            )
                .into_response();
        }
    };

    if !state.config.classifier_routing.enabled {
        return next.run(request).await;
    }

    let Some(classifier) = &state.minilm_classifier else {
        tracing::warn!("Classifier routing enabled but MiniLM classifier is unavailable");
        if let Some(response) = fail_closed_response(&state, "classifier artifact unavailable") {
            return response;
        }
        return next.run(request).await;
    };

    let body_bytes = match request.extensions().get::<BufferedBody>() {
        Some(buf) => buf.0.clone(),
        None => {
            tracing::error!("Classifier routing: BufferedBody missing from request extensions");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Firewall misconfiguration: missing buffered body",
            )
                .into_response();
        }
    };

    let prompt = extract_classifier_context(&body_bytes);
    if prompt.trim().is_empty() {
        if let Some(response) =
            fail_closed_response(&state, "request did not contain classifier context")
        {
            return response;
        }
        return next.run(request).await;
    }

    let classifier = classifier.clone();
    let embedder = state.text_embedder.clone();
    let prompt_for_classification = prompt.clone();
    let classification = match tokio::task::spawn_blocking(move || {
        let embedding = embedder.generate_embedding(&prompt_for_classification)?;
        classifier.classify_embedding(&embedding)
    })
    .await
    {
        Ok(Ok(classification)) => classification,
        Ok(Err(error)) => {
            tracing::warn!(error = %error, "Classifier routing failed — falling back to existing routing");
            if let Some(response) = fail_closed_response(&state, "classification failed") {
                return response;
            }
            return next.run(request).await;
        }
        Err(error) => {
            tracing::warn!(error = %error, "Classifier routing task failed — falling back to existing routing");
            if let Some(response) = fail_closed_response(&state, "classification task failed") {
                return response;
            }
            return next.run(request).await;
        }
    };

    request.extensions_mut().insert(classification.clone());

    let mut matched_route = false;
    if let Some(route) = select_route(&state.config.classifier_routing, &classification) {
        matched_route = true;
        if let Some(model) = route.selected_model.as_deref() {
            request
                .extensions_mut()
                .insert(BufferedBody(Bytes::from(override_request_model(
                    &body_bytes,
                    model,
                ))));
        }
        log_route_selection(&classification, &route);
        request.extensions_mut().insert(route);
    } else {
        tracing::debug!(
            task_type = %classification.task_type.label,
            complexity = %classification.complexity.label,
            persona = %classification.persona.label,
            domain = %classification.domain.label,
            confidence = classification.overall_confidence,
            "Classifier routing produced no matching rule"
        );
    }

    if matched_route || state.config.classifier_routing.fallback_to_existing_routing {
        next.run(request).await
    } else {
        tracing::warn!(
            task_type = %classification.task_type.label,
            complexity = %classification.complexity.label,
            persona = %classification.persona.label,
            domain = %classification.domain.label,
            confidence = classification.overall_confidence,
            "Classifier routing required a rule match but none was produced"
        );
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Classifier routing is required but no matching route was produced",
        )
            .into_response()
    }
}

fn log_route_selection(classification: &MultiHeadClassification, route: &ClassifierRouteDecision) {
    tracing::info!(
        task_type = %classification.task_type.label,
        complexity = %classification.complexity.label,
        persona = %classification.persona.label,
        domain = %classification.domain.label,
        confidence = classification.overall_confidence,
        selected_provider = route.selected_provider.as_deref().unwrap_or("primary"),
        selected_model = route.selected_model.as_deref().unwrap_or("(unchanged)"),
        matched_rule = route.matched_rule_name.as_deref().unwrap_or("(unnamed)"),
        "Classifier routing selected route"
    );
}

fn fail_closed_response(state: &AppState, reason: &'static str) -> Option<Response> {
    if state.config.classifier_routing.fallback_to_existing_routing {
        None
    } else {
        tracing::warn!(reason = reason, "Classifier routing failed closed");
        Some(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Classifier routing is required but {reason}"),
            )
                .into_response(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, middleware as axum_mw, routing::post};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::classifier::{
        ClassificationHeadArtifact, MiniLmMultiHeadClassifier, MultiHeadClassifierArtifact,
    };
    use crate::config::{AppConfig, ClassifierRoutingConfig, ClassifierRoutingRuleConfig};

    async fn read_buffered_model(request: Request) -> Response {
        let body = request
            .extensions()
            .get::<BufferedBody>()
            .map(|body| body.0.clone())
            .unwrap_or_default();
        (StatusCode::OK, body).into_response()
    }

    fn fixture_classifier() -> MiniLmMultiHeadClassifier {
        MiniLmMultiHeadClassifier::from_path(write_temp_artifact()).unwrap()
    }

    fn write_temp_artifact() -> std::path::PathBuf {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.into_temp_path().keep().unwrap();
        let artifact = MultiHeadClassifierArtifact {
            version: 1,
            embedding_dim: 384,
            task_type: ClassificationHeadArtifact {
                labels: vec!["lookup".into(), "codegen".into()],
                weights: vec![vec![0.0; 384], vec![1.0; 384]],
                bias: vec![0.0, 0.0],
            },
            complexity: ClassificationHeadArtifact {
                labels: vec!["simple".into(), "complex".into()],
                weights: vec![vec![0.0; 384], vec![1.0; 384]],
                bias: vec![0.0, 0.0],
            },
            persona: ClassificationHeadArtifact {
                labels: vec!["analyst".into(), "builder".into()],
                weights: vec![vec![0.0; 384], vec![1.0; 384]],
                bias: vec![0.0, 0.0],
            },
            domain: ClassificationHeadArtifact {
                labels: vec!["docs".into(), "code".into()],
                weights: vec![vec![0.0; 384], vec![1.0; 384]],
                bias: vec![0.0, 0.0],
            },
        };
        std::fs::write(&path, serde_json::to_vec(&artifact).unwrap()).unwrap();
        path
    }

    #[tokio::test]
    async fn classifier_routing_overrides_model_in_buffered_body() {
        let mut config = AppConfig::test_default();
        config.classifier_routing = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.0,
            fallback_to_existing_routing: true,
            rules: vec![ClassifierRoutingRuleConfig {
                name: "codegen-route".into(),
                task_type: Some("codegen".into()),
                complexity: Some("complex".into()),
                persona: Some("builder".into()),
                domain: Some("code".into()),
                provider: Some("groq".into()),
                model: Some("llama-3.3-70b-versatile".into()),
                min_confidence: None,
            }],
            matrix: Default::default(),
        };

        let mut state = AppState::new(
            Arc::new(config),
            crate::layer1::embeddings::shared_test_embedder(),
        );
        state.minilm_classifier = Some(Arc::new(fixture_classifier()));

        let app = Router::new()
            .route("/", post(read_buffered_model))
            .layer(axum_mw::from_fn(classifier_routing_middleware))
            .layer(axum_mw::from_fn(
                crate::middleware::body_buffer::buffer_body_middleware,
            ))
            .layer(axum_mw::from_fn(
                move |mut req: Request, next: axum_mw::Next| {
                    let state = Arc::new(state.clone());
                    async move {
                        req.extensions_mut().insert(state);
                        next.run(req).await
                    }
                },
            ));

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "model": "gpt-4o-mini",
                    "messages": [{"role":"user","content":"write rust middleware"}]
                }))
                .unwrap(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["model"], "llama-3.3-70b-versatile");
    }

    #[tokio::test]
    async fn classifier_routing_fails_closed_when_no_matching_rule() {
        let mut config = AppConfig::test_default();
        config.classifier_routing = ClassifierRoutingConfig {
            enabled: true,
            artifacts_path: String::new(),
            confidence_threshold: 0.0,
            fallback_to_existing_routing: false,
            rules: vec![ClassifierRoutingRuleConfig {
                name: "docs-only".into(),
                task_type: Some("lookup".into()),
                complexity: Some("simple".into()),
                persona: Some("analyst".into()),
                domain: Some("docs".into()),
                provider: Some("openai".into()),
                model: Some("gpt-4o-mini".into()),
                min_confidence: None,
            }],
            matrix: Default::default(),
        };

        let mut state = AppState::new(
            Arc::new(config),
            crate::layer1::embeddings::shared_test_embedder(),
        );
        state.minilm_classifier = Some(Arc::new(fixture_classifier()));

        let app = Router::new()
            .route("/", post(read_buffered_model))
            .layer(axum_mw::from_fn(classifier_routing_middleware))
            .layer(axum_mw::from_fn(
                crate::middleware::body_buffer::buffer_body_middleware,
            ))
            .layer(axum_mw::from_fn(
                move |mut req: Request, next: axum_mw::Next| {
                    let state = Arc::new(state.clone());
                    async move {
                        req.extensions_mut().insert(state);
                        next.run(req).await
                    }
                },
            ));

        let req = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "model": "gpt-4o-mini",
                    "messages": [{"role":"user","content":"write rust middleware"}]
                }))
                .unwrap(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
