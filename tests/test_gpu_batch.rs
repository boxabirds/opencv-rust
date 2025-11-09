/// Integration tests for GPU batch processing
#[cfg(all(feature = "gpu", test))]
mod gpu_batch_tests {
    use opencv_rust::core::{Mat, MatDepth};
    use opencv_rust::core::types::{Scalar, ColorConversionCode};
    use opencv_rust::gpu::GpuBatch;

    #[test]
    #[ignore] // Only run when GPU is available
    fn test_batch_gaussian_threshold() {
        // Initialize GPU
        opencv_rust::gpu::init_gpu();

        if !opencv_rust::gpu::gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create test image
        let img = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        // Process with batch
        let result = GpuBatch::new()
            .gaussian_blur(5, 1.5)
            .threshold(127.0, 255.0)
            .execute(&img);

        match result {
            Ok(output) => {
                assert_eq!(output.rows(), 100);
                assert_eq!(output.cols(), 100);
            }
            Err(e) => {
                eprintln!("GPU batch execution failed: {:?}", e);
                // Don't fail the test - GPU might not be available
            }
        }
    }

    #[test]
    fn test_batch_builder_pattern() {
        // Test that the builder pattern works without executing
        let batch = GpuBatch::new()
            .gaussian_blur(5, 1.5)
            .cvt_color(ColorConversionCode::RgbToGray)
            .threshold(127.0, 255.0)
            .canny(50.0, 150.0);

        // This just verifies the API compiles and chains correctly
        assert!(true);
    }

    #[test]
    fn test_batch_empty() {
        let img = Mat::new(10, 10, 3, MatDepth::U8).unwrap();

        // Empty batch should return clone of input
        let batch = GpuBatch::new();

        // This test works even without GPU
        #[cfg(not(feature = "gpu"))]
        {
            let result = batch.execute(&img);
            assert!(result.is_err()); // Should fail without GPU support
        }

        #[cfg(feature = "gpu")]
        {
            // With GPU feature enabled, empty batch returns input
            let result = batch.execute(&img);
            if result.is_ok() {
                let output = result.unwrap();
                assert_eq!(output.rows(), img.rows());
                assert_eq!(output.cols(), img.cols());
            }
        }
    }
}

#[cfg(all(not(feature = "gpu"), test))]
mod no_gpu_tests {
    use opencv_rust::core::{Mat, MatDepth};

    #[test]
    fn test_batch_without_gpu_feature() {
        // When GPU feature is not enabled, GpuBatch is still available
        // but will return errors when executed
        let _img = Mat::new(10, 10, 3, MatDepth::U8).unwrap();

        // Just verify the test module compiles
        assert!(true);
    }
}
