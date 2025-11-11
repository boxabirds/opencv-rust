//! Macros for WASM backend selection
//!
//! These macros eliminate code duplication when adding GPU/CPU backend selection
//! to WASM operations.

/// Simple backend dispatch macro - generates the match statement inline
///
/// This works around borrow checker issues by generating code inline rather than
/// using closures.
///
/// # Example
/// ```ignore
/// backend_dispatch! {
///     gpu => {
///         crate::gpu::ops::canny_gpu_async(&gray, &mut dst, t1, t2).await?;
///     }
///     cpu => {
///         crate::imgproc::canny(&gray, &mut dst, t1, t2)?;
///     }
/// }
/// ```
#[macro_export]
macro_rules! backend_dispatch {
    (gpu => $gpu_block:block cpu => $cpu_block:block) => {
        match $crate::wasm::backend::get_backend() {
            1 => {
                // GPU backend selected
                #[cfg(feature = "gpu")]
                $gpu_block

                #[cfg(not(feature = "gpu"))]
                {
                    return Err(wasm_bindgen::JsValue::from_str(
                        "GPU not available in this build. Use setBackend('cpu')"
                    ));
                }
            }
            _ => {
                // CPU backend
                $cpu_block
            }
        }
    };
}

/// Execute an operation with automatic GPU/CPU backend selection
///
/// # Usage
///
/// For operations WITH GPU implementation:
/// ```ignore
/// backend_dispatch_gpu!(
///     operation_name,
///     gpu::ops::operation_gpu_async,
///     cpu_module::operation,
///     &src.inner,
///     &mut dst,
///     param1,
///     param2
/// );
/// ```
///
/// For operations WITHOUT GPU implementation (CPU-only, future-proof):
/// ```ignore
/// backend_dispatch_cpu_only!(
///     operation_name,
///     cpu_module::operation,
///     &src.inner,
///     &mut dst,
///     param1,
///     param2
/// );
/// ```
#[macro_export]
macro_rules! backend_dispatch_gpu {
    ($op_name:expr, $gpu_fn:path, $cpu_fn:path, $src:expr, $dst:expr $(, $param:expr)*) => {
        match $crate::wasm::backend::get_backend() {
            1 => {
                // GPU backend selected
                #[cfg(feature = "gpu")]
                {
                    $gpu_fn($src, $dst, $($param),*)
                        .await
                        .map_err(|e| {
                            wasm_bindgen::JsValue::from_str(&format!(
                                "GPU {} error: {}. Try setBackend('auto') or setBackend('cpu')",
                                $op_name, e
                            ))
                        })?;
                    return Ok($crate::wasm::WasmMat { inner: $dst.clone() });
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(wasm_bindgen::JsValue::from_str(&format!(
                        "GPU not available for {}. Try setBackend('cpu')",
                        $op_name
                    )));
                }
            }
            _ => {
                // CPU backend (fallback)
                $cpu_fn($src, $dst, $($param),*)
                    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
            }
        }
    };
}

/// Execute a CPU-only operation with backend selection pattern (future-proof)
///
/// This macro provides the same backend selection pattern but only has a CPU path.
/// When GPU implementation becomes available, just switch to `backend_dispatch_gpu!`
#[macro_export]
macro_rules! backend_dispatch_cpu_only {
    ($op_name:expr, $cpu_fn:path, $src:expr, $dst:expr $(, $param:expr)*) => {
        match $crate::wasm::backend::get_backend() {
            1 => {
                // GPU requested but not available
                #[cfg(feature = "gpu")]
                {
                    return Err(wasm_bindgen::JsValue::from_str(&format!(
                        "GPU not yet implemented for {}. Using CPU fallback",
                        $op_name
                    )));
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(wasm_bindgen::JsValue::from_str(&format!(
                        "GPU not available. Using CPU for {}",
                        $op_name
                    )));
                }
            }
            _ => {
                // CPU backend
                $cpu_fn($src, $dst, $($param),*)
                    .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
            }
        }
    };
}

/// Helper macro for creating a destination Mat
#[macro_export]
macro_rules! create_dst_mat {
    ($src:expr) => {
        $crate::core::Mat::new(
            $src.inner.rows(),
            $src.inner.cols(),
            $src.inner.channels(),
            $src.inner.depth(),
        )
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?
    };
    ($src:expr, $depth:expr) => {
        $crate::core::Mat::new(
            $src.inner.rows(),
            $src.inner.cols(),
            $src.inner.channels(),
            $depth,
        )
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?
    };
}
