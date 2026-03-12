// =============================================================================
// Library root — re-exports modules so that integration tests (in tests/)
// can reference them via `use isartor::...`.
//
// The binary entry-point remains in main.rs.
// =============================================================================

pub mod adapters;
pub mod clients;
pub mod config;
pub mod core;
pub mod errors;
pub mod factory;
pub mod handler;
pub mod layer1;
pub mod metrics;
pub mod middleware;
pub mod models;
pub mod pipeline;
#[cfg(feature = "embedded-inference")]
pub mod services;
pub mod state;
pub mod telemetry;
pub mod vector_cache;
