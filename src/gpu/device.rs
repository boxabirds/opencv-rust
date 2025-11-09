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
    pub fn init() -> Option<&'static GpuContext> {
        GPU_CONTEXT.get_or_init(|| {
            pollster::block_on(Self::init_async())
        }).as_ref()
    }

    /// Initialize GPU context asynchronously (works for WASM and native)
    pub async fn init_async() -> Option<GpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("OpenCV-Rust GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .ok()?;

        Some(GpuContext {
            device,
            queue,
            adapter,
        })
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
