/// Pipeline Cache - Reuse GPU compute pipelines across operations
///
/// Creating GPU pipelines is expensive (10-100ms). This cache stores
/// initialized pipelines and reuses them, providing massive performance gains.
///
/// Usage:
/// ```rust
/// // Initialize once (automatically done with GpuContext)
/// PipelineCache::init(&device).await;
///
/// // Get cached pipeline (fast - no recreation)
/// let pipeline = PipelineCache::get_threshold_pipeline();
/// ```

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use wgpu;

#[cfg(feature = "gpu")]
/// Cached pipeline components for a single operation
pub struct CachedPipeline {
    pub shader: wgpu::ShaderModule,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub compute_pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "gpu")]
/// Cached GPU compute pipelines for all core operations
pub struct PipelineCache {
    // Core image processing operations (Batch 1 - Priority)
    pub gaussian_blur: Option<CachedPipeline>,
    pub resize: Option<CachedPipeline>,
    pub threshold: Option<CachedPipeline>,
    pub canny: Option<CachedPipeline>,
    pub sobel: Option<CachedPipeline>,

    // Morphological operations
    pub erode: Option<CachedPipeline>,
    pub dilate: Option<CachedPipeline>,

    // Color conversions
    pub rgb_to_gray: Option<CachedPipeline>,
    pub rgb_to_hsv: Option<CachedPipeline>,
    pub hsv_to_rgb: Option<CachedPipeline>,

    // Advanced filters
    pub bilateral_filter: Option<CachedPipeline>,
    pub median_blur: Option<CachedPipeline>,
    pub adaptive_threshold: Option<CachedPipeline>,
    pub laplacian: Option<CachedPipeline>,
    pub scharr: Option<CachedPipeline>,

    // Geometric transforms
    pub warp_affine: Option<CachedPipeline>,
    pub warp_perspective: Option<CachedPipeline>,
    pub flip: Option<CachedPipeline>,
    pub rotate: Option<CachedPipeline>,

    // Additional operations (lower priority but commonly used)
    pub box_blur: Option<CachedPipeline>,

    // Dynamic pipelines with varying parameters (LRU cache)
    dynamic_cache: HashMap<String, Arc<wgpu::ComputePipeline>>,
    dynamic_cache_max_size: usize,
}

// For native: use OnceLock (requires Send + Sync)
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use std::sync::OnceLock;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
static PIPELINE_CACHE: OnceLock<Option<PipelineCache>> = OnceLock::new();

// For WASM: use thread_local (doesn't require Send + Sync)
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
use std::cell::RefCell;

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
thread_local! {
    static PIPELINE_CACHE: RefCell<Option<PipelineCache>> = RefCell::new(None);
}

#[cfg(feature = "gpu")]
impl PipelineCache {
    /// Create a new empty pipeline cache
    fn new() -> Self {
        PipelineCache {
            gaussian_blur: None,
            resize: None,
            threshold: None,
            canny: None,
            sobel: None,
            erode: None,
            dilate: None,
            rgb_to_gray: None,
            rgb_to_hsv: None,
            hsv_to_rgb: None,
            bilateral_filter: None,
            median_blur: None,
            adaptive_threshold: None,
            laplacian: None,
            scharr: None,
            warp_affine: None,
            warp_perspective: None,
            flip: None,
            rotate: None,
            box_blur: None,
            dynamic_cache: HashMap::new(),
            dynamic_cache_max_size: 100,
        }
    }

    /// Initialize the pipeline cache with pre-compiled pipelines (async)
    /// This should be called once during GPU context initialization
    pub async fn init_async(device: &wgpu::Device) -> bool {
        let mut cache = Self::new();

        // Pre-compile core operations (most frequently used)
        cache.threshold = Self::create_threshold_pipeline(device).await;
        cache.resize = Self::create_resize_pipeline(device).await;
        cache.sobel = Self::create_sobel_pipeline(device).await;
        cache.rgb_to_gray = Self::create_rgb_to_gray_pipeline(device).await;

        // Compile additional operations (commonly used)
        cache.erode = Self::create_erode_pipeline(device).await;
        cache.dilate = Self::create_dilate_pipeline(device).await;
        cache.flip = Self::create_flip_pipeline(device).await;
        cache.laplacian = Self::create_laplacian_pipeline(device).await;

        // Note: gaussian_blur uses separable filters with two entry points (horizontal/vertical)
        // and is compiled on-demand rather than cached

        // Store the cache
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = PIPELINE_CACHE.set(Some(cache));
        }

        #[cfg(target_arch = "wasm32")]
        {
            PIPELINE_CACHE.with(|c| {
                *c.borrow_mut() = Some(cache);
            });
        }

        true
    }

    /// Initialize the pipeline cache synchronously (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn init(device: &wgpu::Device) -> bool {
        if PIPELINE_CACHE.get().is_some() {
            return PIPELINE_CACHE.get().unwrap().is_some();
        }
        pollster::block_on(Self::init_async(device))
    }

    /// Get the cached threshold pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_threshold_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .threshold
            .as_ref()
    }

    /// Get the cached threshold pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_threshold_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .threshold
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached gaussian blur pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_gaussian_blur_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .gaussian_blur
            .as_ref()
    }

    /// Get the cached gaussian blur pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_gaussian_blur_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .gaussian_blur
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached resize pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_resize_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .resize
            .as_ref()
    }

    /// Get the cached resize pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_resize_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .resize
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached sobel pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_sobel_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .sobel
            .as_ref()
    }

    /// Get the cached sobel pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_sobel_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .sobel
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached RGB to grayscale pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_rgb_to_gray_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .rgb_to_gray
            .as_ref()
    }

    /// Get the cached RGB to grayscale pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_rgb_to_gray_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .rgb_to_gray
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached flip pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_flip_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .flip
            .as_ref()
    }

    /// Get the cached flip pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_flip_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .flip
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached erode pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_erode_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .erode
            .as_ref()
    }

    /// Get the cached erode pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_erode_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .erode
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached dilate pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_dilate_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .dilate
            .as_ref()
    }

    /// Get the cached dilate pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_dilate_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .dilate
                .as_ref()
                .map(f)
        })
    }

    /// Get the cached laplacian pipeline
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_laplacian_pipeline() -> Option<&'static CachedPipeline> {
        PIPELINE_CACHE
            .get()?
            .as_ref()?
            .laplacian
            .as_ref()
    }

    /// Get the cached laplacian pipeline (WASM)
    #[cfg(target_arch = "wasm32")]
    pub fn with_laplacian_pipeline<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        PIPELINE_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()?
                .laplacian
                .as_ref()
                .map(f)
        })
    }

    // Pipeline creation functions
    // These create the actual pipeline objects with shaders, layouts, etc.

    async fn create_threshold_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Threshold Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/threshold.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Threshold Bind Group Layout"),
            entries: &[
                // Input buffer (binding 0)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer (binding 1)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params buffer (binding 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Threshold Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Threshold Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("threshold_binary"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_gaussian_blur_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gaussian Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/gaussian_blur.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Blur Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Blur Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gaussian Blur Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gaussian_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_resize_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resize Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/resize.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Resize Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resize Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Resize Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("resize_bilinear"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_sobel_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sobel Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/sobel.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sobel Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sobel Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sobel Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sobel_filter"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_rgb_to_gray_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RGB to Gray Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/rgb_to_gray.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RGB to Gray Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RGB to Gray Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RGB to Gray Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("rgb_to_gray"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_erode_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Erode Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/erode.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Erode Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Erode Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Erode Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("erode"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_dilate_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dilate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/dilate.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dilate Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dilate Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dilate Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dilate"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_flip_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flip Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/flip.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flip Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flip Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flip Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("flip"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }

    async fn create_laplacian_pipeline(device: &wgpu::Device) -> Option<CachedPipeline> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Laplacian Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/laplacian.wgsl").into()
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Laplacian Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Laplacian Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Laplacian Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("laplacian"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(CachedPipeline {
            shader,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
        })
    }
}

// No-op implementation for when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
pub struct PipelineCache;

#[cfg(not(feature = "gpu"))]
impl PipelineCache {
    pub fn init() -> bool {
        false
    }

    pub async fn init_async() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_pipeline_cache_structure() {
        let cache = PipelineCache::new();
        assert!(cache.threshold.is_none());
        assert!(cache.gaussian_blur.is_none());
        assert_eq!(cache.dynamic_cache_max_size, 100);
    }
}
