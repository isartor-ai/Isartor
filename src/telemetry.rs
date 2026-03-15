use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    metrics::{MeterProviderBuilder, PeriodicReader, SdkMeterProvider},
    trace::{SdkTracerProvider, TracerProviderBuilder},
    Resource,
};
use opentelemetry_semantic_conventions::resource::{SERVICE_NAME, SERVICE_VERSION};
use std::sync::Arc;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::AppConfig;

// ═══════════════════════════════════════════════════════════════════════
// OtelGuard — RAII flush on shutdown
// ═══════════════════════════════════════════════════════════════════════

/// Holds the OTel providers so they can be flushed and shut down cleanly
/// when the guard is dropped (at the end of `main`).
pub struct OtelGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(tp) = self.tracer_provider.take() {
            if let Err(e) = tp.shutdown() {
                eprintln!("[otel] tracer provider shutdown error: {e:?}");
            }
        }
        if let Some(mp) = self.meter_provider.take() {
            if let Err(e) = mp.shutdown() {
                eprintln!("[otel] meter provider shutdown error: {e:?}");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Initialisation
// ═══════════════════════════════════════════════════════════════════════

/// Initialise structured logging, OpenTelemetry tracing (OTLP/gRPC),
/// and metric collection.
///
/// Returns an [`OtelGuard`] that **must** be held until process exit so
/// that in-flight spans and metrics are flushed cleanly.
///
/// When `enable_monitoring` is `false`, only console logging is active
/// and the guard is inert (no OTel SDK initialised).
pub fn init_telemetry(config: &Arc<AppConfig>) -> anyhow::Result<OtelGuard> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,h2=warn,hyper=warn,tower=warn"));

    // ── Console-only mode ────────────────────────────────────────
    if !config.enable_monitoring {
        // Pretty format for local development.
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().pretty())
            .init();

        tracing::info!("Telemetry: console-only mode (ISARTOR__ENABLE_MONITORING=false)");
        return Ok(OtelGuard {
            tracer_provider: None,
            meter_provider: None,
        });
    }

    // ── Offline mode: disable OTel exporter to prevent phone-home ───
    if config.offline_mode {
        let is_external = !crate::core::is_internal_endpoint(&config.otel_exporter_endpoint);

        if is_external {
            // Fall back to console-only: an external OTel push would be a
            // phone-home violation in an air-gapped environment.
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer().pretty())
                .init();

            tracing::warn!(
                otel.endpoint = %config.otel_exporter_endpoint,
                "Telemetry: OTel exporter DISABLED (offline mode — \
                 external endpoint would escape the perimeter). \
                 Set ISARTOR__OTEL_EXPORTER_ENDPOINT to an internal \
                 collector to enable telemetry in offline mode."
            );
            return Ok(OtelGuard {
                tracer_provider: None,
                meter_provider: None,
            });
        }
    }

    // ── Full OTel mode ───────────────────────────────────────────
    let endpoint = &config.otel_exporter_endpoint;

    let resource = Resource::builder_empty()
        .with_attributes(vec![
            KeyValue::new(SERVICE_NAME, "isartor-gateway"),
            KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
        ])
        .build();

    // 1. Distributed Tracing — TracerProvider → OTLP gRPC
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
    let otel_trace_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // 2. Metrics — MeterProvider → OTLP gRPC → Prometheus-compatible
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

    // 3. tracing-subscriber: structured JSON to stdout + OTel layer
    //    JSON output is required for fluentd / Logstash / CloudWatch ingestion.
    let json_layer = fmt::layer()
        .json()
        .with_target(true)
        .with_thread_ids(false)
        .with_span_list(true)
        .flatten_event(true);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(json_layer)
        .with(otel_trace_layer)
        .init();

    tracing::info!(
        otel.endpoint = %endpoint,
        service.name = "isartor-gateway",
        service.version = env!("CARGO_PKG_VERSION"),
        "Telemetry: OpenTelemetry ENABLED (traces + metrics → OTLP gRPC)"
    );

    Ok(OtelGuard {
        tracer_provider: Some(tracer_provider),
        meter_provider: Some(meter_provider),
    })
}
