/**
 * OpenCV.js Dynamic Loader
 *
 * Loads opencv.js from CDN and provides helper functions to run operations
 * for performance benchmarking against our GPU implementation.
 */

import { useEffect, useState } from 'react';

// OpenCV.js URL - using local copy to avoid CDN/network issues
// Path is relative to project root (Vite config allows serving from parent dirs)
// To use CDN instead: 'https://docs.opencv.org/4.8.0/opencv.js'
const OPENCV_JS_URL = '/cache/opencv.js';

// Global state
let opencvLoaded = false;
let opencvLoading = false;
let cv = null;
const loadCallbacks = [];

/**
 * Load OpenCV.js from CDN
 */
export const loadOpenCVJs = () => {
  return new Promise((resolve, reject) => {
    // Already loaded
    if (opencvLoaded && cv) {
      resolve(cv);
      return;
    }

    // Currently loading - add to callback queue
    if (opencvLoading) {
      loadCallbacks.push({ resolve, reject });
      return;
    }

    // Start loading
    opencvLoading = true;

    const script = document.createElement('script');
    script.src = OPENCV_JS_URL;
    script.async = true;

    script.onerror = (error) => {
      console.error('Failed to load opencv.js:', error);
      opencvLoading = false;
      reject(new Error('Failed to load opencv.js from CDN'));

      // Reject all queued callbacks
      loadCallbacks.forEach(cb => cb.reject(error));
      loadCallbacks.length = 0;
    };

    script.onload = () => {
      // OpenCV.js requires onRuntimeInitialized callback
      if (window.cv) {
        window.cv.onRuntimeInitialized = () => {
          cv = window.cv;
          opencvLoaded = true;
          opencvLoading = false;

          console.log('OpenCV.js loaded successfully');
          resolve(cv);

          // Resolve all queued callbacks
          loadCallbacks.forEach(cb => cb.resolve(cv));
          loadCallbacks.length = 0;
        };
      } else {
        const error = new Error('opencv.js loaded but cv object not found');
        opencvLoading = false;
        reject(error);

        loadCallbacks.forEach(cb => cb.reject(error));
        loadCallbacks.length = 0;
      }
    };

    document.head.appendChild(script);
  });
};

/**
 * Check if OpenCV.js is loaded
 */
export const isOpenCVJsLoaded = () => {
  return opencvLoaded && cv !== null;
};

/**
 * React hook to load OpenCV.js (lazy loading - only when needed)
 * @param {boolean} shouldLoad - Set to true to trigger loading
 */
export const useOpenCVJs = (shouldLoad = false) => {
  const [loaded, setLoaded] = useState(opencvLoaded);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Don't auto-load unless explicitly requested
    if (!shouldLoad) {
      return;
    }

    if (opencvLoaded) {
      setLoaded(true);
      return;
    }

    setLoading(true);
    loadOpenCVJs()
      .then(() => {
        setLoaded(true);
        setLoading(false);
      })
      .catch((err) => {
        setError(err);
        setLoading(false);
      });
  }, [shouldLoad]);

  return { loaded, loading, error, cv };
};

/**
 * Operation mappings: our WASM name -> opencv.js implementation
 */
export const runOpenCVJsOperation = async (operationName, imageData, params) => {
  if (!cv) {
    throw new Error('OpenCV.js not loaded');
  }

  try {
    // Create Mat from ImageData
    const src = cv.matFromImageData(imageData);
    const dst = new cv.Mat();

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
        // Convert to grayscale first
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
        throw new Error(`Operation ${operationName} not implemented for opencv.js`);
    }

    // Convert result back to ImageData
    const resultCanvas = document.createElement('canvas');
    resultCanvas.width = dst.cols;
    resultCanvas.height = dst.rows;
    cv.imshow(resultCanvas, dst);
    const resultImageData = resultCanvas.getContext('2d').getImageData(0, 0, dst.cols, dst.rows);

    // Cleanup
    src.delete();
    dst.delete();

    return resultImageData;
  } catch (error) {
    console.error(`OpenCV.js operation ${operationName} failed:`, error);
    throw error;
  }
};

/**
 * Benchmark an operation with opencv.js
 */
export const benchmarkOpenCVJs = async (operationName, imageData, params, iterations = 10) => {
  if (!cv) {
    throw new Error('OpenCV.js not loaded');
  }

  const timings = [];

  // Warmup
  for (let i = 0; i < 3; i++) {
    await runOpenCVJsOperation(operationName, imageData, params);
  }

  // Benchmark
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await runOpenCVJsOperation(operationName, imageData, params);
    const end = performance.now();
    timings.push(end - start);
  }

  // Calculate statistics
  const sorted = [...timings].sort((a, b) => a - b);
  const mean = timings.reduce((a, b) => a + b, 0) / timings.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const min = sorted[0];
  const max = sorted[sorted.length - 1];

  return {
    mean,
    median,
    min,
    max,
    timings,
  };
};

export default {
  loadOpenCVJs,
  isOpenCVJsLoaded,
  useOpenCVJs,
  runOpenCVJsOperation,
  benchmarkOpenCVJs,
};
