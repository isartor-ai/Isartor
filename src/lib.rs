// =============================================================================
// Library root — re-exports modules so that integration tests (in tests/)
// can reference them via `use isartor::...`.
//
// The binary entry-point remains in main.rs.
// =============================================================================

pub mod adapters;
pub mod anthropic_sse;
pub mod cli;
pub mod clients;
pub mod config;
pub mod core;
pub mod demo;
pub mod errors;
pub mod factory;
pub mod first_run;
pub mod handler;
pub mod health;
pub mod hf;
pub mod layer1;
pub mod metrics;
pub mod middleware;
pub mod models;
pub mod pipeline;
pub mod providers;
pub mod proxy;
#[cfg(feature = "embedded-inference")]
pub mod services;
pub mod state;
pub mod telemetry;
pub mod tool_identity;
pub mod vector_cache;
pub mod visibility;
