/**
 * Bit-Level Verification Tests for opencv-rust vs opencv.js
 *
 * Compares pixel-perfect output between implementations to verify correctness.
 * Uses configurable tolerances for operations with acceptable numerical differences.
 *
 * @version 1.0.0
 * @license Apache-2.0
 */

import tolerancesData from './tolerances.json' assert { type: 'json' };

// Will be set when test runner loads
let cv = null; // opencv.js
let cvRust = null; // opencv-rust compat layer
let wasmModule = null; // opencv-rust WASM bindings

/**
 * Test result statistics
 */
class ComparisonResult {
  constructor(operationName, imageSize, pass = false) {
    this.operationName = operationName;
    this.imageSize = imageSize;
    this.pass = pass;

    // Timing
    this.opencvJsTime = 0;
    this.opencvRustTime = 0;
    this.speedup = 0;

    // Pixel differences
    this.maxDiff = 0;
    this.meanDiff = 0;
    this.rmse = 0;
    this.differentPixels = 0;
    this.percentDifferent = 0;

    // Tolerance
    this.tolerance = tolerancesData.tolerances[operationName] || tolerancesData.defaults;

    // Bit-perfect flag
    this.bitPerfect = false;

    // Failure details
    this.failures = [];
  }

  /**
   * Check if result passes tolerance thresholds
   */
  checkPass() {
    this.pass = true;

    if (this.maxDiff > this.tolerance.maxDiff) {
      this.pass = false;
      this.failures.push(`Max diff ${this.maxDiff} exceeds threshold ${this.tolerance.maxDiff}`);
    }

    if (this.percentDifferent > this.tolerance.percentDifferent) {
      this.pass = false;
      this.failures.push(`Percent different ${this.percentDifferent.toFixed(2)}% exceeds threshold ${this.tolerance.percentDifferent}%`);
    }

    if (this.meanDiff > this.tolerance.meanDiffThreshold) {
      this.pass = false;
      this.failures.push(`Mean diff ${this.meanDiff.toFixed(3)} exceeds threshold ${this.tolerance.meanDiffThreshold}`);
    }

    this.bitPerfect = (this.maxDiff === 0 && this.differentPixels === 0);

    return this.pass;
  }

  /**
   * Format result as readable string
   */
  toString() {
    const icon = this.pass ? '✅' : '❌';
    const perfectIcon = this.bitPerfect ? ' 🎯' : '';

    let output = `\nTesting: ${this.operationName} (${this.imageSize.width}x${this.imageSize.height})\n`;
    output += `├─ opencv.js:     ${this.opencvJsTime.toFixed(1)}ms\n`;
    output += `├─ opencv-rust:   ${this.opencvRustTime.toFixed(1)}ms (${this.speedup.toFixed(1)}x faster)\n`;
    output += `├─ Max diff:      ${this.maxDiff} (threshold: ${this.tolerance.maxDiff}) ${this.maxDiff <= this.tolerance.maxDiff ? '✅' : '❌'}\n`;
    output += `├─ Mean diff:     ${this.meanDiff.toFixed(2)} (threshold: ${this.tolerance.meanDiffThreshold}) ${this.meanDiff <= this.tolerance.meanDiffThreshold ? '✅' : '❌'}\n`;
    output += `├─ RMSE:          ${this.rmse.toFixed(2)}\n`;
    output += `├─ Different px:  ${this.differentPixels} (${this.percentDifferent.toFixed(3)}%) (threshold: ${this.tolerance.percentDifferent}%) ${this.percentDifferent <= this.tolerance.percentDifferent ? '✅' : '❌'}\n`;

    if (this.failures.length > 0) {
      output += `├─ Failures:\n`;
      this.failures.forEach(f => {
        output += `│  └─ ${f}\n`;
      });
    }

    output += `└─ Result: ${this.pass ? 'PASS' : 'FAIL'} ${icon}${perfectIcon}\n`;

    if (this.bitPerfect) {
      output += `   (Bit-perfect match!)\n`;
    }

    return output;
  }

  /**
   * Convert to JSON for reporting
   */
  toJSON() {
    return {
      operationName: this.operationName,
      imageSize: this.imageSize,
      pass: this.pass,
      bitPerfect: this.bitPerfect,
      timing: {
        opencvJs: this.opencvJsTime,
        opencvRust: this.opencvRustTime,
        speedup: this.speedup
      },
      differences: {
        maxDiff: this.maxDiff,
        meanDiff: this.meanDiff,
        rmse: this.rmse,
        differentPixels: this.differentPixels,
        percentDifferent: this.percentDifferent
      },
      tolerance: this.tolerance,
      failures: this.failures
    };
  }
}

/**
 * Compare two ImageData objects pixel by pixel
 * @param {ImageData} imageData1 - First image (opencv.js output)
 * @param {ImageData} imageData2 - Second image (opencv-rust output)
 * @param {number} toleranceThreshold - Pixel difference to count as "different" (default: 1)
 * @returns {ComparisonResult} Statistics about differences
 */
export function comparePixelData(imageData1, imageData2, result, toleranceThreshold = 1) {
  const data1 = imageData1.data;
  const data2 = imageData2.data;

  if (data1.length !== data2.length) {
    throw new Error(`Image data length mismatch: ${data1.length} vs ${data2.length}`);
  }

  if (imageData1.width !== imageData2.width || imageData1.height !== imageData2.height) {
    throw new Error(`Image size mismatch: ${imageData1.width}x${imageData1.height} vs ${imageData2.width}x${imageData2.height}`);
  }

  let maxDiff = 0;
  let sumDiff = 0;
  let sumSquaredDiff = 0;
  let differentPixels = 0;
  const totalPixels = imageData1.width * imageData1.height;

  // Compare each pixel channel
  for (let i = 0; i < data1.length; i++) {
    const diff = Math.abs(data1[i] - data2[i]);

    if (diff > maxDiff) {
      maxDiff = diff;
    }

    sumDiff += diff;
    sumSquaredDiff += diff * diff;

    // Count pixels with any channel difference > threshold
    if (diff > toleranceThreshold && i % 4 !== 3) { // Skip alpha channel
      differentPixels++;
    }
  }

  // Calculate statistics
  const totalChannels = data1.length;
  result.maxDiff = maxDiff;
  result.meanDiff = sumDiff / totalChannels;
  result.rmse = Math.sqrt(sumSquaredDiff / totalChannels);
  result.differentPixels = Math.floor(differentPixels / 3); // Divide by 3 since we count RGB separately
  result.percentDifferent = (result.differentPixels / totalPixels) * 100;

  return result;
}

/**
 * Generate test images
 */
export function generateTestImages() {
  const images = {};

  // 1. Solid color (simple test)
  images.solid = createSolidImage(256, 256, 128, 128, 128);

  // 2. Gradient (tests interpolation)
  images.gradient = createGradientImage(256, 256);

  // 3. Checkerboard (tests edge handling)
  images.checkerboard = createCheckerboardImage(256, 256, 16);

  // 4. Random noise (tests robustness)
  images.noise = createNoiseImage(256, 256);

  // 5. Photo-like (realistic test)
  images.photolike = createPhotLikeImage(512, 512);

  return images;
}

function createSolidImage(width, height, r, g, b) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
  ctx.fillRect(0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

function createGradientImage(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, 'black');
  gradient.addColorStop(0.5, 'gray');
  gradient.addColorStop(1, 'white');

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  return ctx.getImageData(0, 0, width, height);
}

function createCheckerboardImage(width, height, squareSize) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  for (let y = 0; y < height; y += squareSize) {
    for (let x = 0; x < width; x += squareSize) {
      const isWhite = ((x / squareSize) + (y / squareSize)) % 2 === 0;
      ctx.fillStyle = isWhite ? 'white' : 'black';
      ctx.fillRect(x, y, squareSize, squareSize);
    }
  }

  return ctx.getImageData(0, 0, width, height);
}

function createNoiseImage(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);

  for (let i = 0; i < imageData.data.length; i += 4) {
    imageData.data[i] = Math.random() * 255;     // R
    imageData.data[i + 1] = Math.random() * 255; // G
    imageData.data[i + 2] = Math.random() * 255; // B
    imageData.data[i + 3] = 255;                 // A
  }

  ctx.putImageData(imageData, 0, 0);
  return imageData;
}

function createPhotLikeImage(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  // Create a simple "photo-like" scene with shapes
  // Sky gradient
  const skyGradient = ctx.createLinearGradient(0, 0, 0, height / 2);
  skyGradient.addColorStop(0, '#87CEEB'); // Sky blue
  skyGradient.addColorStop(1, '#E0F6FF');
  ctx.fillStyle = skyGradient;
  ctx.fillRect(0, 0, width, height / 2);

  // Ground
  ctx.fillStyle = '#8B7355';
  ctx.fillRect(0, height / 2, width, height / 2);

  // Sun
  ctx.fillStyle = '#FFD700';
  ctx.beginPath();
  ctx.arc(width * 0.8, height * 0.2, 40, 0, Math.PI * 2);
  ctx.fill();

  // Tree (simple)
  ctx.fillStyle = '#654321';
  ctx.fillRect(width * 0.2, height * 0.3, 20, height * 0.4);
  ctx.fillStyle = '#228B22';
  ctx.beginPath();
  ctx.arc(width * 0.21, height * 0.3, 50, 0, Math.PI * 2);
  ctx.fill();

  return ctx.getImageData(0, 0, width, height);
}

/**
 * Test a single operation
 * @param {string} operationName - Name of the operation to test
 * @param {Function} opencvJsOp - Function that runs opencv.js operation
 * @param {Function} opencvRustOp - Function that runs opencv-rust operation
 * @param {ImageData} testImage - Test image to use
 * @returns {Promise<ComparisonResult>} Test result
 */
export async function testOperation(operationName, opencvJsOp, opencvRustOp, testImage) {
  const result = new ComparisonResult(
    operationName,
    { width: testImage.width, height: testImage.height }
  );

  try {
    // Run opencv.js
    const t1 = performance.now();
    const opencvJsOutput = await opencvJsOp(testImage);
    result.opencvJsTime = performance.now() - t1;

    // Run opencv-rust
    const t2 = performance.now();
    const opencvRustOutput = await opencvRustOp(testImage);
    result.opencvRustTime = performance.now() - t2;

    // Calculate speedup
    result.speedup = result.opencvJsTime / result.opencvRustTime;

    // Compare outputs pixel by pixel
    comparePixelData(opencvJsOutput, opencvRustOutput, result);

    // Check if result passes tolerance
    result.checkPass();

  } catch (error) {
    result.pass = false;
    result.failures.push(`Exception: ${error.message}`);
  }

  return result;
}

/**
 * Test suite: Core filtering operations
 */
export async function testFiltering(testImages) {
  const results = [];

  // Gaussian Blur
  results.push(await testOperation(
    'gaussianBlur',
    async (img) => {
      const src = cv.matFromImageData(img);
      const dst = new cv.Mat();
      cv.GaussianBlur(src, dst, new cv.Size(5, 5), 1.5);
      const output = new ImageData(new Uint8ClampedArray(dst.data), dst.cols, dst.rows);
      src.delete();
      dst.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const dst = new cvRust.Mat();
      await cvRust.GaussianBlur(src, dst, {width: 5, height: 5}, 1.5);
      return dst.toImageData();
    },
    testImages.gradient
  ));

  // Median Blur
  results.push(await testOperation(
    'medianBlur',
    async (img) => {
      const src = cv.matFromImageData(img);
      const dst = new cv.Mat();
      cv.medianBlur(src, dst, 5);
      const output = new ImageData(new Uint8ClampedArray(dst.data), dst.cols, dst.rows);
      src.delete();
      dst.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const dst = new cvRust.Mat();
      await cvRust.medianBlur(src, dst, 5);
      return dst.toImageData();
    },
    testImages.noise
  ));

  // Bilateral Filter
  results.push(await testOperation(
    'bilateralFilter',
    async (img) => {
      const src = cv.matFromImageData(img);
      const dst = new cv.Mat();
      cv.bilateralFilter(src, dst, 9, 75, 75);
      const output = new ImageData(new Uint8ClampedArray(dst.data), dst.cols, dst.rows);
      src.delete();
      dst.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const dst = new cvRust.Mat();
      await cvRust.bilateralFilter(src, dst, 9, 75, 75);
      return dst.toImageData();
    },
    testImages.photolike
  ));

  return results;
}

/**
 * Test suite: Edge detection
 */
export async function testEdgeDetection(testImages) {
  const results = [];

  // Canny
  results.push(await testOperation(
    'canny',
    async (img) => {
      const src = cv.matFromImageData(img);
      const gray = new cv.Mat();
      const edges = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.Canny(gray, edges, 50, 150);
      const output = new ImageData(new Uint8ClampedArray(edges.data), edges.cols, edges.rows);
      src.delete();
      gray.delete();
      edges.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const gray = new cvRust.Mat();
      const edges = new cvRust.Mat();
      await cvRust.cvtColor(src, gray, cvRust.COLOR_RGB2GRAY);
      await cvRust.Canny(gray, edges, 50, 150);
      return edges.toImageData();
    },
    testImages.photolike
  ));

  // Sobel
  results.push(await testOperation(
    'sobel',
    async (img) => {
      const src = cv.matFromImageData(img);
      const gray = new cv.Mat();
      const sobel = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.Sobel(gray, sobel, cv.CV_8U, 1, 0, 3);
      const output = new ImageData(new Uint8ClampedArray(sobel.data), sobel.cols, sobel.rows);
      src.delete();
      gray.delete();
      sobel.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const gray = new cvRust.Mat();
      const sobel = new cvRust.Mat();
      await cvRust.cvtColor(src, gray, cvRust.COLOR_RGB2GRAY);
      await cvRust.Sobel(gray, sobel, -1, 1, 0, 3);
      return sobel.toImageData();
    },
    testImages.gradient
  ));

  return results;
}

/**
 * Test suite: Thresholding
 */
export async function testThresholding(testImages) {
  const results = [];

  // Binary Threshold
  results.push(await testOperation(
    'threshold',
    async (img) => {
      const src = cv.matFromImageData(img);
      const gray = new cv.Mat();
      const binary = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.threshold(gray, binary, 127, 255, cv.THRESH_BINARY);
      const output = new ImageData(new Uint8ClampedArray(binary.data), binary.cols, binary.rows);
      src.delete();
      gray.delete();
      binary.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const gray = new cvRust.Mat();
      const binary = new cvRust.Mat();
      await cvRust.cvtColor(src, gray, cvRust.COLOR_RGB2GRAY);
      await cvRust.threshold(gray, binary, 127, 255, cvRust.THRESH_BINARY);
      return binary.toImageData();
    },
    testImages.gradient
  ));

  // Adaptive Threshold
  results.push(await testOperation(
    'adaptiveThreshold',
    async (img) => {
      const src = cv.matFromImageData(img);
      const gray = new cv.Mat();
      const binary = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.adaptiveThreshold(gray, binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);
      const output = new ImageData(new Uint8ClampedArray(binary.data), binary.cols, binary.rows);
      src.delete();
      gray.delete();
      binary.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const gray = new cvRust.Mat();
      const binary = new cvRust.Mat();
      await cvRust.cvtColor(src, gray, cvRust.COLOR_RGB2GRAY);
      await cvRust.adaptiveThreshold(gray, binary, 255, cvRust.ADAPTIVE_THRESH_GAUSSIAN_C, cvRust.THRESH_BINARY, 11, 2);
      return binary.toImageData();
    },
    testImages.photolike
  ));

  return results;
}

/**
 * Test suite: Color conversions
 */
export async function testColorConversions(testImages) {
  const results = [];

  // RGB to Gray
  results.push(await testOperation(
    'cvtColorGray',
    async (img) => {
      const src = cv.matFromImageData(img);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      const output = new ImageData(new Uint8ClampedArray(gray.data), gray.cols, gray.rows);
      src.delete();
      gray.delete();
      return output;
    },
    async (img) => {
      const src = new cvRust.Mat(0, 0, 0, img);
      const gray = new cvRust.Mat();
      await cvRust.cvtColor(src, gray, cvRust.COLOR_RGB2GRAY);
      return gray.toImageData();
    },
    testImages.photolike
  ));

  return results;
}

/**
 * Run all test suites
 */
export async function runAllTests() {
  console.log('='.repeat(60));
  console.log('OpenCV.js vs opencv-rust: Bit-Level Verification Tests');
  console.log('='.repeat(60));

  const testImages = generateTestImages();
  const allResults = [];

  console.log('\n📸 Generated test images:');
  console.log(`  - solid: ${testImages.solid.width}x${testImages.solid.height}`);
  console.log(`  - gradient: ${testImages.gradient.width}x${testImages.gradient.height}`);
  console.log(`  - checkerboard: ${testImages.checkerboard.width}x${testImages.checkerboard.height}`);
  console.log(`  - noise: ${testImages.noise.width}x${testImages.noise.height}`);
  console.log(`  - photolike: ${testImages.photolike.width}x${testImages.photolike.height}`);

  // Run test suites
  console.log('\n🔬 Running Filtering Tests...');
  const filterResults = await testFiltering(testImages);
  allResults.push(...filterResults);
  filterResults.forEach(r => console.log(r.toString()));

  console.log('\n🔬 Running Edge Detection Tests...');
  const edgeResults = await testEdgeDetection(testImages);
  allResults.push(...edgeResults);
  edgeResults.forEach(r => console.log(r.toString()));

  console.log('\n🔬 Running Thresholding Tests...');
  const threshResults = await testThresholding(testImages);
  allResults.push(...threshResults);
  threshResults.forEach(r => console.log(r.toString()));

  console.log('\n🔬 Running Color Conversion Tests...');
  const colorResults = await testColorConversions(testImages);
  allResults.push(...colorResults);
  colorResults.forEach(r => console.log(r.toString()));

  // Summary
  const passed = allResults.filter(r => r.pass).length;
  const failed = allResults.filter(r => !r.pass).length;
  const bitPerfect = allResults.filter(r => r.bitPerfect).length;
  const avgSpeedup = allResults.reduce((sum, r) => sum + r.speedup, 0) / allResults.length;

  console.log('\n' + '='.repeat(60));
  console.log('📊 SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total tests:      ${allResults.length}`);
  console.log(`Passed:           ${passed} ✅`);
  console.log(`Failed:           ${failed} ❌`);
  console.log(`Bit-perfect:      ${bitPerfect} 🎯`);
  console.log(`Avg speedup:      ${avgSpeedup.toFixed(2)}x`);
  console.log(`Success rate:     ${((passed / allResults.length) * 100).toFixed(1)}%`);

  if (failed > 0) {
    console.log('\n⚠️  Failed tests:');
    allResults.filter(r => !r.pass).forEach(r => {
      console.log(`  - ${r.operationName}`);
      r.failures.forEach(f => console.log(`    └─ ${f}`));
    });
  }

  return {
    allResults,
    summary: {
      total: allResults.length,
      passed,
      failed,
      bitPerfect,
      avgSpeedup,
      successRate: (passed / allResults.length) * 100
    }
  };
}

/**
 * Initialize test environment
 */
export function initTests(opencvJs, opencvRust, wasm) {
  cv = opencvJs;
  cvRust = opencvRust;
  wasmModule = wasm;
}

// Export for use in HTML test runner
if (typeof window !== 'undefined') {
  window.BitLevelTests = {
    initTests,
    runAllTests,
    testOperation,
    comparePixelData,
    generateTestImages,
    ComparisonResult
  };
}
