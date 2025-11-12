import { test, expect } from '@playwright/test';
import { compareImages, formatComparisonReport } from './pixel_comparison.js';
import { TEST_CONFIGS } from './generate_tests.js';

/**
 * OpenCV.js Parity Test: Gaussian Blur
 *
 * Compares our WASM gaussian_blur implementation against opencv.js
 * reference using pixel-level comparison.
 *
 * Tolerance: max_pixel_diff=1, max_mean_diff=0.1, max_outliers=1%
 * Rationale: Floating-point rounding in convolution kernels may
 * cause minor differences (±1px), but mean should be very close.
 */

const config = TEST_CONFIGS.gaussian_blur;
const TEST_IMAGES = ['lenna.png'];

test.describe('Gaussian Blur - OpenCV.js Parity', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to test harness
    await page.goto('/tests/opencv_js_reference/test-harness.html');

    // Wait for both libraries to load
    await page.waitForFunction(() => window.testHarnessReady, {
      timeout: 60000,
    });
  });

  // Test each parameter configuration
  for (const params of config.params) {
    for (const imageName of TEST_IMAGES) {
      test(`gaussian_blur ksize=${params.ksize} sigma=${params.sigma} on ${imageName}`, async ({
        page,
      }) => {
        console.log(
          `\nTesting gaussian_blur with ksize=${params.ksize}, sigma=${params.sigma} on ${imageName}`
        );

        // Run comparison in browser
        const result = await page.evaluate(
          async ({ imageName, ksize, sigma }) => {
            // Load test image
            const img = await window.loadTestImage(imageName);

            // Create OpenCV.js Mat from ImageData
            const cvSrc = window.imageDataToMat(img);

            // ===== Run OpenCV.js (reference) =====
            const cvDst = new cv.Mat();
            const cvStart = performance.now();

            cv.GaussianBlur(
              cvSrc,
              cvDst,
              new cv.Size(ksize, ksize),
              sigma,
              sigma,
              cv.BORDER_DEFAULT
            );

            const cvTime = performance.now() - cvStart;
            const cvResult = window.matToImageData(cvDst);

            // ===== Run our WASM implementation =====
            const wasmStart = performance.now();

            // Create our Mat from image data
            const wasmSrc = window.opencvRust.WasmMat.fromImageData(
              new Uint8Array(img.data),
              img.width,
              img.height,
              4  // RGBA = 4 channels
            );

            // Run gaussian blur
            const wasmDst = await window.opencvRust.gaussianBlur(
              wasmSrc,
              ksize,
              sigma
            );

            const wasmTime = performance.now() - wasmStart;

            // Get result data
            const wasmData = wasmDst.getData();

            // Clean up
            cvSrc.delete();
            cvDst.delete();
            wasmSrc.free();
            wasmDst.free();

            return {
              opencv: {
                data: Array.from(cvResult.data),
                width: cvResult.width,
                height: cvResult.height,
                time: cvTime,
              },
              ours: {
                data: Array.from(wasmData),
                width: img.width,
                height: img.height,
                time: wasmTime,
              },
            };
          },
          { imageName, ksize: params.ksize, sigma: params.sigma }
        );

        // Compare results pixel-by-pixel
        const comparison = compareImages(
          { data: result.opencv.data },
          { data: result.ours.data },
          config.tolerance
        );

        // Log results
        console.log('\nPerformance:');
        console.log(`  OpenCV.js: ${result.opencv.time.toFixed(2)}ms`);
        console.log(`  Our WASM:  ${result.ours.time.toFixed(2)}ms`);
        console.log(
          `  Speedup:   ${(result.opencv.time / result.ours.time).toFixed(2)}x`
        );

        console.log('\n' + formatComparisonReport(comparison));

        // Assert
        expect(comparison.passed, formatComparisonReport(comparison)).toBe(
          true
        );

        // Additional sanity checks
        expect(comparison.maxDiff).toBeLessThanOrEqual(
          config.tolerance.max_pixel_diff * 2
        );
        expect(comparison.meanDiff).toBeLessThanOrEqual(
          config.tolerance.max_mean_diff
        );
        expect(comparison.outlierPercent).toBeLessThanOrEqual(
          config.tolerance.max_outliers_percent
        );
      });
    }
  }
});

test.describe('Gaussian Blur - Edge Cases', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/tests/opencv_js_reference/test-harness.html');
    await page.waitForFunction(() => window.testHarnessReady, {
      timeout: 60000,
    });
  });

  test('handles small kernel size (ksize=3)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const img = await window.loadTestImage('lenna.png');
      const cvSrc = window.imageDataToMat(img);

      // OpenCV.js
      const cvDst = new cv.Mat();
      cv.GaussianBlur(
        cvSrc,
        cvDst,
        new cv.Size(3, 3),
        0, // sigma=0 means auto-calculate
        0,
        cv.BORDER_DEFAULT
      );
      const cvResult = window.matToImageData(cvDst);

      // Our WASM
      const wasmSrc = window.opencvRust.WasmMat.fromImageData(
        new Uint8Array(img.data),
        img.width,
        img.height,
        4  // RGBA = 4 channels
      );
      const wasmDst = await window.opencvRust.gaussianBlur(wasmSrc, 3, 0);
      const wasmData = wasmDst.getData();

      cvSrc.delete();
      cvDst.delete();
      wasmSrc.free();
      wasmDst.free();

      return {
        opencv: Array.from(cvResult.data),
        ours: Array.from(wasmData),
      };
    });

    const comparison = compareImages(
      { data: result.opencv },
      { data: result.ours },
      config.tolerance
    );

    expect(comparison.passed).toBe(true);
  });

  test('handles large kernel size (ksize=21)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const img = await window.loadTestImage('lenna.png');
      const cvSrc = window.imageDataToMat(img);

      // OpenCV.js
      const cvDst = new cv.Mat();
      cv.GaussianBlur(
        cvSrc,
        cvDst,
        new cv.Size(21, 21),
        5.0,
        5.0,
        cv.BORDER_DEFAULT
      );
      const cvResult = window.matToImageData(cvDst);

      // Our WASM
      const wasmSrc = window.opencvRust.WasmMat.fromImageData(
        new Uint8Array(img.data),
        img.width,
        img.height,
        4  // RGBA = 4 channels
      );
      const wasmDst = await window.opencvRust.gaussianBlur(wasmSrc, 21, 5.0);
      const wasmData = wasmDst.getData();

      cvSrc.delete();
      cvDst.delete();
      wasmSrc.free();
      wasmDst.free();

      return {
        opencv: Array.from(cvResult.data),
        ours: Array.from(wasmData),
      };
    });

    const comparison = compareImages(
      { data: result.opencv },
      { data: result.ours },
      config.tolerance
    );

    expect(comparison.passed).toBe(true);
  });
});
