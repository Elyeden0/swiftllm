// SwiftLLM library crate — shared between the CLI binary and the Python extension.

pub mod config;
pub mod consensus;
pub mod endpoints;
pub mod failover;
pub mod ffi;
pub mod middleware;
pub mod providers;
pub mod routing;
pub mod server;

#[cfg(feature = "python")]
pub mod python;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export the PyO3 module init function so maturin can find it.
#[cfg(feature = "python")]
pub use python::_swiftllm;
