use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    metrics::{MeterProviderBuilder, PeriodicReader},
    trace::TracerProviderBuilder,
    Resource,
};
use opentelemetry_semantic_conventions::resource::SERVICE_NAME;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::AppConfig;

/// Initialise structured logging, OpenTelemetry tracing (OTLP/gRPC),
/// and metric collection.  When `enable_monitoring` is `false` only
/// console logging is active.
pub fn init_telemetry(config: Arc<AppConfig>) -> anyhow::Result<()> {
    let resource = Resource::builder_empty()
        .with_attributes(vec![KeyValue::new(SERVICE_NAME, "isartor-gateway")])
        .build();

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().pretty();

    if !config.enable_monitoring {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
        tracing::info!("OpenTelemetry monitoring disabled. Using local console logs.");
        return Ok(());
    }

    let endpoint = &config.otel_exporter_endpoint;

    // ── 1. Distributed Tracing (TracerProvider → OTLP gRPC) ──────
    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let tracer_provider = TracerProviderBuilder::default()
        .with_batch_exporter(span_exporter)
        .with_resource(resource.clone())
        .build();

    global::set_tracer_provider(tracer_provider.clone());
    let tracer = global::tracer("isartor-gateway");
    let tracer_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // ── 2. Metrics (MeterProvider → OTLP gRPC → Prometheus) ──────
    let metrics_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let reader = PeriodicReader::builder(metrics_exporter).build();

    let meter_provider = MeterProviderBuilder::default()
        .with_resource(resource)
        .with_reader(reader)
        .build();

    global::set_meter_provider(meter_provider.clone());

    // ── 3. tracing-subscriber: console + OTel layer ──────────────
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(tracer_layer)
        .init();

    tracing::info!(
        endpoint = %endpoint,
        "OpenTelemetry monitoring ENABLED (Traces & Metrics)."
    );

    Ok(())
}

/// Flush in-flight spans & metrics before process exit.
#[allow(dead_code)]
pub fn shutdown_telemetry() {
    // In OpenTelemetry 0.31+, shutdown is handled by dropping the
    // TracerProvider. We replace the global provider with a no-op,
    // which causes the original provider to drop and flush.
    opentelemetry::global::set_tracer_provider(
        opentelemetry::trace::noop::NoopTracerProvider::new(),
    );
}
