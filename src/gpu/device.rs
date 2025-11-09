#[cfg(feature = "gpu")]
use wgpu;

#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

// For native: use OnceLock (requires Send + Sync)
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use std::sync::OnceLock;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

// For WASM: use thread_local (doesn't require Send + Sync)
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
use std::cell::RefCell;

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
thread_local! {
    static GPU_CONTEXT: RefCell<Option<GpuContext>> = RefCell::new(None);
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Initialize GPU context synchronously (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn init() -> bool {
        if GPU_CONTEXT.get().is_some() {
            return GPU_CONTEXT.get().unwrap().is_some();
        }
        pollster::block_on(Self::init_async())
    }

    /// Initialize GPU context asynchronously (works for WASM and native)
    /// Returns true if initialization succeeded
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn init_async() -> bool {
        // Check if already initialized
        if GPU_CONTEXT.get().is_some() {
            return GPU_CONTEXT.get().unwrap().is_some();
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) => a,
            None => {
                let _ = GPU_CONTEXT.set(None);
                return false;
            }
        };

        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("OpenCV-Rust GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
        {
            Ok(dq) => dq,
            Err(_) => {
                let _ = GPU_CONTEXT.set(None);
                return false;
            }
        };

        let ctx = GpuContext {
            device,
            queue,
            adapter,
        };

        // Store in global context
        let _ = GPU_CONTEXT.set(Some(ctx));
        true
    }

    /// Initialize GPU context asynchronously for WASM
    #[cfg(target_arch = "wasm32")]
    pub async fn init_async() -> bool {
        // Check if already initialized
        let already_init = GPU_CONTEXT.with(|ctx| ctx.borrow().is_some());
        if already_init {
            return true;
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) => a,
            None => {
                GPU_CONTEXT.with(|ctx| *ctx.borrow_mut() = None);
                return false;
            }
        };

        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("OpenCV-Rust GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
        {
            Ok(dq) => dq,
            Err(_) => {
                GPU_CONTEXT.with(|ctx| *ctx.borrow_mut() = None);
                return false;
            }
        };

        let ctx = GpuContext {
            device,
            queue,
            adapter,
        };

        // Store in thread-local context
        GPU_CONTEXT.with(|context| *context.borrow_mut() = Some(ctx));
        true
    }

    /// Get the global GPU context if initialized (native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get() -> Option<&'static GpuContext> {
        GPU_CONTEXT.get()?.as_ref()
    }

    /// Get reference to GPU context for WASM - requires closure due to RefCell
    #[cfg(target_arch = "wasm32")]
    pub fn with_context<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&GpuContext) -> R,
    {
        GPU_CONTEXT.with(|ctx| {
            let borrowed = ctx.borrow();
            borrowed.as_ref().map(f)
        })
    }

    /// Check if GPU is available (native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_available() -> bool {
        GPU_CONTEXT.get().map(|ctx| ctx.is_some()).unwrap_or(false)
    }

    /// Check if GPU is available (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn is_available() -> bool {
        GPU_CONTEXT.with(|ctx| ctx.borrow().is_some())
    }

    /// Execute a function with GPU context - works for both native and WASM
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_gpu<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&GpuContext) -> R,
    {
        Self::get().map(f)
    }

    /// Execute a function with GPU context - works for both native and WASM
    #[cfg(target_arch = "wasm32")]
    pub fn with_gpu<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&GpuContext) -> R,
    {
        Self::with_context(f)
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn is_available() -> bool {
        false
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn get() -> Option<&'static GpuContext> {
        None
    }
}
