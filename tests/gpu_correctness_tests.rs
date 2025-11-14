// Auto-generated GPU tests
#![cfg(all(test, feature = "gpu"))]

#[cfg(all(test, feature = "gpu"))]
mod farneback_optical_flow_gpu_tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::core::types::Scalar;

    #[test]
    fn test_farneback_optical_flow_gpu_vs_cpu() {
        // Initialize GPU
        if !GpuContext::init() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create test image
        let width = 640;
        let height = 480;
        let channels = 4;
        let test_img = Mat::new_with_default(
            height, width, channels,
            MatDepth::U8,
            Scalar::new(128.0, 128.0, 128.0, 255.0)
        ).unwrap();

        // Run CPU version
        let cpu_result = farneback_optical_flow_cpu(&test_img).unwrap();

        // Run GPU version
        let gpu_result = futures::executor::block_on(
            farneback_optical_flow_gpu(GpuContext::get().unwrap(), &test_img)
        ).unwrap();

        // Compare results
        assert_eq!(cpu_result.rows(), gpu_result.rows());
        assert_eq!(cpu_result.cols(), gpu_result.cols());
        assert_eq!(cpu_result.channels(), gpu_result.channels());

        // Pixel-level comparison (allow small floating point differences)
        let tolerance = 2; // Max difference in pixel values
        let mut diff_count = 0;
        let total_pixels = (cpu_result.rows() * cpu_result.cols() * cpu_result.channels()) as usize;

        for i in 0..total_pixels {
            let cpu_val = cpu_result.data()[i];
            let gpu_val = gpu_result.data()[i];
            let diff = (cpu_val as i32 - gpu_val as i32).abs();
            if diff > tolerance {
                diff_count += 1;
            }
        }

        let diff_percent = (diff_count as f32 / total_pixels as f32) * 100.0;
        assert!(diff_percent < 1.0,
            "farneback_optical_flow: {}% pixels differ by more than {} (expected <1%)",
            diff_percent, tolerance
        );
    }

    #[test]
    fn test_farneback_optical_flow_gpu_performance() {
        if !GpuContext::init() {
            return;
        }

        let test_img = Mat::new_with_default(
            1080, 1920, 4,
            MatDepth::U8,
            Scalar::all(128.0)
        ).unwrap();

        // Warmup
        let _ = futures::executor::block_on(
            farneback_optical_flow_gpu(GpuContext::get().unwrap(), &test_img)
        );

        // Benchmark GPU
        let gpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = futures::executor::block_on(
                farneback_optical_flow_gpu(GpuContext::get().unwrap(), &test_img)
            ).unwrap();
        }
        let gpu_time = gpu_start.elapsed().as_millis() / 10;

        // Benchmark CPU
        let cpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = farneback_optical_flow_cpu(&test_img).unwrap();
        }
        let cpu_time = cpu_start.elapsed().as_millis() / 10;

        let speedup = cpu_time as f32 / gpu_time as f32;
        println!("farneback_optical_flow: GPU={}ms, CPU={}ms, Speedup={:.2}x",
                 gpu_time, cpu_time, speedup);

        // GPU should be faster for this operation
        assert!(speedup > 1.0, "GPU should be faster than CPU");
    }
}


#[cfg(all(test, feature = "gpu"))]
mod nlm_denoising_gpu_tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::core::types::Scalar;

    #[test]
    fn test_nlm_denoising_gpu_vs_cpu() {
        // Initialize GPU
        if !GpuContext::init() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create test image
        let width = 640;
        let height = 480;
        let channels = 4;
        let test_img = Mat::new_with_default(
            height, width, channels,
            MatDepth::U8,
            Scalar::new(128.0, 128.0, 128.0, 255.0)
        ).unwrap();

        // Run CPU version
        let cpu_result = nlm_denoising_cpu(&test_img).unwrap();

        // Run GPU version
        let gpu_result = futures::executor::block_on(
            nlm_denoising_gpu(GpuContext::get().unwrap(), &test_img)
        ).unwrap();

        // Compare results
        assert_eq!(cpu_result.rows(), gpu_result.rows());
        assert_eq!(cpu_result.cols(), gpu_result.cols());
        assert_eq!(cpu_result.channels(), gpu_result.channels());

        // Pixel-level comparison (allow small floating point differences)
        let tolerance = 2; // Max difference in pixel values
        let mut diff_count = 0;
        let total_pixels = (cpu_result.rows() * cpu_result.cols() * cpu_result.channels()) as usize;

        for i in 0..total_pixels {
            let cpu_val = cpu_result.data()[i];
            let gpu_val = gpu_result.data()[i];
            let diff = (cpu_val as i32 - gpu_val as i32).abs();
            if diff > tolerance {
                diff_count += 1;
            }
        }

        let diff_percent = (diff_count as f32 / total_pixels as f32) * 100.0;
        assert!(diff_percent < 1.0,
            "nlm_denoising: {}% pixels differ by more than {} (expected <1%)",
            diff_percent, tolerance
        );
    }

    #[test]
    fn test_nlm_denoising_gpu_performance() {
        if !GpuContext::init() {
            return;
        }

        let test_img = Mat::new_with_default(
            1080, 1920, 4,
            MatDepth::U8,
            Scalar::all(128.0)
        ).unwrap();

        // Warmup
        let _ = futures::executor::block_on(
            nlm_denoising_gpu(GpuContext::get().unwrap(), &test_img)
        );

        // Benchmark GPU
        let gpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = futures::executor::block_on(
                nlm_denoising_gpu(GpuContext::get().unwrap(), &test_img)
            ).unwrap();
        }
        let gpu_time = gpu_start.elapsed().as_millis() / 10;

        // Benchmark CPU
        let cpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = nlm_denoising_cpu(&test_img).unwrap();
        }
        let cpu_time = cpu_start.elapsed().as_millis() / 10;

        let speedup = cpu_time as f32 / gpu_time as f32;
        println!("nlm_denoising: GPU={}ms, CPU={}ms, Speedup={:.2}x",
                 gpu_time, cpu_time, speedup);

        // GPU should be faster for this operation
        assert!(speedup > 1.0, "GPU should be faster than CPU");
    }
}


#[cfg(all(test, feature = "gpu"))]
mod kmeans_gpu_tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::core::types::Scalar;

    #[test]
    fn test_kmeans_gpu_vs_cpu() {
        // Initialize GPU
        if !GpuContext::init() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create test image
        let width = 640;
        let height = 480;
        let channels = 4;
        let test_img = Mat::new_with_default(
            height, width, channels,
            MatDepth::U8,
            Scalar::new(128.0, 128.0, 128.0, 255.0)
        ).unwrap();

        // Run CPU version
        let cpu_result = kmeans_cpu(&test_img).unwrap();

        // Run GPU version
        let gpu_result = futures::executor::block_on(
            kmeans_gpu(GpuContext::get().unwrap(), &test_img)
        ).unwrap();

        // Compare results
        assert_eq!(cpu_result.rows(), gpu_result.rows());
        assert_eq!(cpu_result.cols(), gpu_result.cols());
        assert_eq!(cpu_result.channels(), gpu_result.channels());

        // Pixel-level comparison (allow small floating point differences)
        let tolerance = 2; // Max difference in pixel values
        let mut diff_count = 0;
        let total_pixels = (cpu_result.rows() * cpu_result.cols() * cpu_result.channels()) as usize;

        for i in 0..total_pixels {
            let cpu_val = cpu_result.data()[i];
            let gpu_val = gpu_result.data()[i];
            let diff = (cpu_val as i32 - gpu_val as i32).abs();
            if diff > tolerance {
                diff_count += 1;
            }
        }

        let diff_percent = (diff_count as f32 / total_pixels as f32) * 100.0;
        assert!(diff_percent < 1.0,
            "kmeans: {}% pixels differ by more than {} (expected <1%)",
            diff_percent, tolerance
        );
    }

    #[test]
    fn test_kmeans_gpu_performance() {
        if !GpuContext::init() {
            return;
        }

        let test_img = Mat::new_with_default(
            1080, 1920, 4,
            MatDepth::U8,
            Scalar::all(128.0)
        ).unwrap();

        // Warmup
        let _ = futures::executor::block_on(
            kmeans_gpu(GpuContext::get().unwrap(), &test_img)
        );

        // Benchmark GPU
        let gpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = futures::executor::block_on(
                kmeans_gpu(GpuContext::get().unwrap(), &test_img)
            ).unwrap();
        }
        let gpu_time = gpu_start.elapsed().as_millis() / 10;

        // Benchmark CPU
        let cpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = kmeans_cpu(&test_img).unwrap();
        }
        let cpu_time = cpu_start.elapsed().as_millis() / 10;

        let speedup = cpu_time as f32 / gpu_time as f32;
        println!("kmeans: GPU={}ms, CPU={}ms, Speedup={:.2}x",
                 gpu_time, cpu_time, speedup);

        // GPU should be faster for this operation
        assert!(speedup > 1.0, "GPU should be faster than CPU");
    }
}


#[cfg(all(test, feature = "gpu"))]
mod gabor_filter_gpu_tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::core::types::Scalar;

    #[test]
    fn test_gabor_filter_gpu_vs_cpu() {
        // Initialize GPU
        if !GpuContext::init() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create test image
        let width = 640;
        let height = 480;
        let channels = 4;
        let test_img = Mat::new_with_default(
            height, width, channels,
            MatDepth::U8,
            Scalar::new(128.0, 128.0, 128.0, 255.0)
        ).unwrap();

        // Run CPU version
        let cpu_result = gabor_filter_cpu(&test_img).unwrap();

        // Run GPU version
        let gpu_result = futures::executor::block_on(
            gabor_filter_gpu(GpuContext::get().unwrap(), &test_img)
        ).unwrap();

        // Compare results
        assert_eq!(cpu_result.rows(), gpu_result.rows());
        assert_eq!(cpu_result.cols(), gpu_result.cols());
        assert_eq!(cpu_result.channels(), gpu_result.channels());

        // Pixel-level comparison (allow small floating point differences)
        let tolerance = 2; // Max difference in pixel values
        let mut diff_count = 0;
        let total_pixels = (cpu_result.rows() * cpu_result.cols() * cpu_result.channels()) as usize;

        for i in 0..total_pixels {
            let cpu_val = cpu_result.data()[i];
            let gpu_val = gpu_result.data()[i];
            let diff = (cpu_val as i32 - gpu_val as i32).abs();
            if diff > tolerance {
                diff_count += 1;
            }
        }

        let diff_percent = (diff_count as f32 / total_pixels as f32) * 100.0;
        assert!(diff_percent < 1.0,
            "gabor_filter: {}% pixels differ by more than {} (expected <1%)",
            diff_percent, tolerance
        );
    }

    #[test]
    fn test_gabor_filter_gpu_performance() {
        if !GpuContext::init() {
            return;
        }

        let test_img = Mat::new_with_default(
            1080, 1920, 4,
            MatDepth::U8,
            Scalar::all(128.0)
        ).unwrap();

        // Warmup
        let _ = futures::executor::block_on(
            gabor_filter_gpu(GpuContext::get().unwrap(), &test_img)
        );

        // Benchmark GPU
        let gpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = futures::executor::block_on(
                gabor_filter_gpu(GpuContext::get().unwrap(), &test_img)
            ).unwrap();
        }
        let gpu_time = gpu_start.elapsed().as_millis() / 10;

        // Benchmark CPU
        let cpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = gabor_filter_cpu(&test_img).unwrap();
        }
        let cpu_time = cpu_start.elapsed().as_millis() / 10;

        let speedup = cpu_time as f32 / gpu_time as f32;
        println!("gabor_filter: GPU={}ms, CPU={}ms, Speedup={:.2}x",
                 gpu_time, cpu_time, speedup);

        // GPU should be faster for this operation
        assert!(speedup > 1.0, "GPU should be faster than CPU");
    }
}


#[cfg(all(test, feature = "gpu"))]
mod guided_filter_gpu_tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::core::types::Scalar;

    #[test]
    fn test_guided_filter_gpu_vs_cpu() {
        // Initialize GPU
        if !GpuContext::init() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create test image
        let width = 640;
        let height = 480;
        let channels = 4;
        let test_img = Mat::new_with_default(
            height, width, channels,
            MatDepth::U8,
            Scalar::new(128.0, 128.0, 128.0, 255.0)
        ).unwrap();

        // Run CPU version
        let cpu_result = guided_filter_cpu(&test_img).unwrap();

        // Run GPU version
        let gpu_result = futures::executor::block_on(
            guided_filter_gpu(GpuContext::get().unwrap(), &test_img)
        ).unwrap();

        // Compare results
        assert_eq!(cpu_result.rows(), gpu_result.rows());
        assert_eq!(cpu_result.cols(), gpu_result.cols());
        assert_eq!(cpu_result.channels(), gpu_result.channels());

        // Pixel-level comparison (allow small floating point differences)
        let tolerance = 2; // Max difference in pixel values
        let mut diff_count = 0;
        let total_pixels = (cpu_result.rows() * cpu_result.cols() * cpu_result.channels()) as usize;

        for i in 0..total_pixels {
            let cpu_val = cpu_result.data()[i];
            let gpu_val = gpu_result.data()[i];
            let diff = (cpu_val as i32 - gpu_val as i32).abs();
            if diff > tolerance {
                diff_count += 1;
            }
        }

        let diff_percent = (diff_count as f32 / total_pixels as f32) * 100.0;
        assert!(diff_percent < 1.0,
            "guided_filter: {}% pixels differ by more than {} (expected <1%)",
            diff_percent, tolerance
        );
    }

    #[test]
    fn test_guided_filter_gpu_performance() {
        if !GpuContext::init() {
            return;
        }

        let test_img = Mat::new_with_default(
            1080, 1920, 4,
            MatDepth::U8,
            Scalar::all(128.0)
        ).unwrap();

        // Warmup
        let _ = futures::executor::block_on(
            guided_filter_gpu(GpuContext::get().unwrap(), &test_img)
        );

        // Benchmark GPU
        let gpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = futures::executor::block_on(
                guided_filter_gpu(GpuContext::get().unwrap(), &test_img)
            ).unwrap();
        }
        let gpu_time = gpu_start.elapsed().as_millis() / 10;

        // Benchmark CPU
        let cpu_start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = guided_filter_cpu(&test_img).unwrap();
        }
        let cpu_time = cpu_start.elapsed().as_millis() / 10;

        let speedup = cpu_time as f32 / gpu_time as f32;
        println!("guided_filter: GPU={}ms, CPU={}ms, Speedup={:.2}x",
                 gpu_time, cpu_time, speedup);

        // GPU should be faster for this operation
        assert!(speedup > 1.0, "GPU should be faster than CPU");
    }
}


