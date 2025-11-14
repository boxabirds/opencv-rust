/**
 * OpenCV.js Web Worker
 *
 * Loads and runs OpenCV.js in a separate thread to prevent main thread blocking.
 * Communicates with main thread via postMessage.
 */

// Import OpenCV.js
self.importScripts('/cache/opencv.js');

let cv = null;
let isInitialized = false;

// Initialize OpenCV.js
const initializeOpenCV = () => {
  return new Promise((resolve, reject) => {
    if (isInitialized && cv) {
      resolve();
      return;
    }

    console.log('[Worker] Loading OpenCV.js...');

    if (!self.cv) {
      reject(new Error('OpenCV.js not loaded'));
      return;
    }

    // Set timeout for initialization
    const timeout = setTimeout(() => {
      reject(new Error('OpenCV.js initialization timeout (>30s)'));
    }, 30000);

    self.cv.onRuntimeInitialized = () => {
      clearTimeout(timeout);
      cv = self.cv;
      isInitialized = true;
      console.log('[Worker] ✓ OpenCV.js initialized successfully');
      resolve();
    };
  });
};

// Run an OpenCV operation
const runOperation = (operationName, imageData, params) => {
  if (!cv) {
    throw new Error('OpenCV.js not initialized');
  }

  // Create Mat from ImageData
  const src = cv.matFromImageData(imageData);
  const dst = new cv.Mat();

  try {
    // Execute operation based on name
    switch (operationName) {
      case 'gaussian_blur': {
        const { ksize = 5, sigma = 1.5 } = params;
        const kSize = new cv.Size(ksize, ksize);
        cv.GaussianBlur(src, dst, kSize, sigma, sigma, cv.BORDER_DEFAULT);
        break;
      }

      case 'resize': {
        const { width, height } = params;
        const dsize = new cv.Size(width, height);
        cv.resize(src, dst, dsize, 0, 0, cv.INTER_LINEAR);
        break;
      }

      case 'threshold': {
        const { thresh = 128, maxval = 255, type = cv.THRESH_BINARY } = params;
        cv.threshold(src, dst, thresh, maxval, type);
        break;
      }

      case 'canny': {
        const { threshold1 = 50, threshold2 = 150 } = params;
        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.Canny(gray, dst, threshold1, threshold2);
        gray.delete();
        break;
      }

      case 'sobel': {
        const { dx = 1, dy = 0, ksize = 3 } = params;
        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.Sobel(gray, dst, cv.CV_8U, dx, dy, ksize);
        gray.delete();
        break;
      }

      case 'erode': {
        const { ksize = 5, iterations = 1 } = params;
        const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(ksize, ksize));
        cv.erode(src, dst, kernel, new cv.Point(-1, -1), iterations);
        kernel.delete();
        break;
      }

      case 'dilate': {
        const { ksize = 5, iterations = 1 } = params;
        const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(ksize, ksize));
        cv.dilate(src, dst, kernel, new cv.Point(-1, -1), iterations);
        kernel.delete();
        break;
      }

      case 'bilateral_filter': {
        const { d = 9, sigmaColor = 75, sigmaSpace = 75 } = params;
        cv.bilateralFilter(src, dst, d, sigmaColor, sigmaSpace);
        break;
      }

      case 'median_blur': {
        const { ksize = 5 } = params;
        cv.medianBlur(src, dst, ksize);
        break;
      }

      case 'laplacian': {
        const { ksize = 1 } = params;
        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.Laplacian(gray, dst, cv.CV_8U, ksize);
        gray.delete();
        break;
      }

      case 'flip': {
        const { flipCode = 1 } = params;
        cv.flip(src, dst, flipCode);
        break;
      }

      case 'cvt_color_gray': {
        cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
        // Convert back to RGBA for display
        const rgba = new cv.Mat();
        cv.cvtColor(dst, rgba, cv.COLOR_GRAY2RGBA);
        dst.delete();
        dst.data = rgba.data;
        rgba.delete();
        break;
      }

      case 'adaptive_threshold': {
        const { maxValue = 255, blockSize = 11, C = 2 } = params;
        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.adaptiveThreshold(
          gray,
          dst,
          maxValue,
          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY,
          blockSize,
          C
        );
        gray.delete();
        // Convert back to RGBA
        const rgba = new cv.Mat();
        cv.cvtColor(dst, rgba, cv.COLOR_GRAY2RGBA);
        dst.delete();
        dst.data = rgba.data;
        rgba.delete();
        break;
      }

      default:
        throw new Error(`Operation ${operationName} not implemented`);
    }

    // Convert result to ImageData
    const resultImageData = new ImageData(
      new Uint8ClampedArray(dst.data),
      dst.cols,
      dst.rows
    );

    // Cleanup
    src.delete();
    dst.delete();

    return resultImageData;
  } catch (error) {
    src.delete();
    dst.delete();
    throw error;
  }
};

// Message handler
self.onmessage = async (event) => {
  const { type, id, operation, imageData, params } = event.data;

  try {
    switch (type) {
      case 'init':
        await initializeOpenCV();
        self.postMessage({ type: 'init_success', id });
        break;

      case 'run_operation':
        const result = runOperation(operation, imageData, params);
        self.postMessage({
          type: 'operation_result',
          id,
          result
        }, [result.data.buffer]); // Transfer ownership of ArrayBuffer
        break;

      case 'benchmark':
        const { iterations = 10 } = params;
        const timings = [];

        // Warmup
        for (let i = 0; i < 3; i++) {
          runOperation(operation, imageData, params);
        }

        // Benchmark
        for (let i = 0; i < iterations; i++) {
          const start = performance.now();
          runOperation(operation, imageData, params);
          const end = performance.now();
          timings.push(end - start);
        }

        // Calculate statistics
        const sorted = [...timings].sort((a, b) => a - b);
        const mean = timings.reduce((a, b) => a + b, 0) / timings.length;
        const median = sorted[Math.floor(sorted.length / 2)];
        const min = sorted[0];
        const max = sorted[sorted.length - 1];

        self.postMessage({
          type: 'benchmark_result',
          id,
          result: { mean, median, min, max, timings }
        });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      id,
      error: error.message
    });
  }
};

// Ready signal
console.log('[Worker] OpenCV.js worker ready');
