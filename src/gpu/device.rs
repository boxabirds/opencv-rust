#[cfg(feature = "gpu")]
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
use wgpu;

#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

#[cfg(feature = "gpu")]
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

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

    /// Get the global GPU context if initialized
    pub fn get() -> Option<&'static GpuContext> {
        GPU_CONTEXT.get()?.as_ref()
    }

    /// Check if GPU is available
    pub fn is_available() -> bool {
        GPU_CONTEXT.get().map(|ctx| ctx.is_some()).unwrap_or(false)
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn is_available() -> bool {
        false
    }

    pub fn get() -> Option<&'static GpuContext> {
        None
    }
}
