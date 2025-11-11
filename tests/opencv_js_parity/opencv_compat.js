/**
 * OpenCV.js Compatibility Layer for opencv-rust WASM
 *
 * This file provides an opencv.js-compatible API wrapper around our async WASM bindings.
 * Use this to migrate existing opencv.js code with minimal changes.
 *
 * Usage:
 *   import * as cv from './opencv_compat.js';
 *   // Now use cv.* functions as you would with opencv.js
 *
 * Key Differences:
 * 1. All operations are async (return Promises)
 * 2. GPU acceleration available via cv.setBackend('webgpu')
 * 3. Mat wrapper uses WasmMat internally
 *
 * @version 1.0.0
 * @license Apache-2.0
 */

import init, * as wasmBindings from './pkg/opencv_rust_wasm.js';

// Initialize WASM module
let initialized = false;
async function ensureInit() {
  if (!initialized) {
    await init();
    initialized = true;
  }
}

// ============================================================================
// Core Types & Constants
// ============================================================================

/**
 * Mat wrapper compatible with opencv.js cv.Mat
 * Internally uses WasmMat but provides opencv.js-like interface
 */
export class Mat {
  constructor(rows, cols, type, data) {
    if (arguments.length === 0) {
      // Empty constructor
      this._wasmMat = null;
    } else if (data instanceof ImageData) {
      // From ImageData
      this._wasmMat = wasmBindings.WasmMat.fromImageData(data);
    } else {
      // TODO: Support other constructors (rows, cols, type, data array)
      throw new Error('Mat constructor not fully implemented');
    }
  }

  /**
   * Get underlying WasmMat (internal use)
   */
  _getWasmMat() {
    return this._wasmMat;
  }

  /**
   * Set underlying WasmMat (internal use)
   */
  _setWasmMat(wasmMat) {
    this._wasmMat = wasmMat;
  }

  /**
   * Convert to ImageData (for display)
   */
  toImageData() {
    if (!this._wasmMat) return null;
    return this._wasmMat.getData();
  }

  /**
   * Get data as typed array (opencv.js compatibility)
   */
  data() {
    const imageData = this.toImageData();
    return imageData ? imageData.data : new Uint8ClampedArray();
  }

  /**
   * Delete/free the Mat (opencv.js compatibility)
   */
  delete() {
    if (this._wasmMat) {
      // WasmMat is garbage collected automatically in Rust/WASM
      this._wasmMat = null;
    }
  }

  /**
   * Get number of rows
   */
  rows() {
    // TODO: Implement if WasmMat exposes dimensions
    return 0;
  }

  /**
   * Get number of cols
   */
  cols() {
    // TODO: Implement if WasmMat exposes dimensions
    return 0;
  }

  /**
   * Get number of channels
   */
  channels() {
    // TODO: Implement if WasmMat exposes channels
    return 4; // Assume RGBA for now
  }

  /**
   * Clone the Mat
   */
  clone() {
    // TODO: Implement clone
    throw new Error('Mat.clone() not implemented');
  }
}

// OpenCV Constants - Threshold Types
export const THRESH_BINARY = 0;
export const THRESH_BINARY_INV = 1;
export const THRESH_TRUNC = 2;
export const THRESH_TOZERO = 3;
export const THRESH_TOZERO_INV = 4;
export const THRESH_OTSU = 8;
export const THRESH_TRIANGLE = 16;

// Adaptive Threshold Types
export const ADAPTIVE_THRESH_MEAN_C = 0;
export const ADAPTIVE_THRESH_GAUSSIAN_C = 1;

// Color Conversion Codes
export const COLOR_BGR2GRAY = 6;
export const COLOR_RGB2GRAY = 7;
export const COLOR_BGR2HSV = 40;
export const COLOR_RGB2HSV = 41;
export const COLOR_HSV2BGR = 54;
export const COLOR_HSV2RGB = 55;
export const COLOR_BGR2Lab = 44;
export const COLOR_RGB2Lab = 45;
export const COLOR_Lab2BGR = 56;
export const COLOR_Lab2RGB = 57;
export const COLOR_BGR2YCrCb = 36;
export const COLOR_RGB2YCrCb = 37;
export const COLOR_YCrCb2BGR = 38;
export const COLOR_YCrCb2RGB = 39;
export const COLOR_BGR2RGB = 4;
export const COLOR_RGB2BGR = 4;
export const COLOR_BGR2XYZ = 32;
export const COLOR_RGB2XYZ = 33;
export const COLOR_XYZ2BGR = 34;
export const COLOR_XYZ2RGB = 35;

// Morphology Types
export const MORPH_ERODE = 0;
export const MORPH_DILATE = 1;
export const MORPH_OPEN = 2;
export const MORPH_CLOSE = 3;
export const MORPH_GRADIENT = 4;
export const MORPH_TOPHAT = 5;
export const MORPH_BLACKHAT = 6;

// Morphology Shapes
export const MORPH_RECT = 0;
export const MORPH_CROSS = 1;
export const MORPH_ELLIPSE = 2;

// Interpolation Flags
export const INTER_NEAREST = 0;
export const INTER_LINEAR = 1;
export const INTER_CUBIC = 2;
export const INTER_AREA = 3;
export const INTER_LANCZOS4 = 4;

// Flip Codes
export const FLIP_HORIZONTAL = 1;
export const FLIP_VERTICAL = 0;
export const FLIP_BOTH = -1;

// ============================================================================
// Backend Management (opencv-rust specific)
// ============================================================================

/**
 * Initialize GPU backend (opencv-rust specific)
 * @returns {Promise<boolean>} True if GPU initialized successfully
 */
export async function initGpu() {
  await ensureInit();
  return await wasmBindings.initGpu();
}

/**
 * Set backend: 'auto', 'webgpu', or 'cpu' (opencv-rust specific)
 * @param {string} backend - Backend to use
 */
export function setBackend(backend) {
  wasmBindings.setBackend(backend);
}

/**
 * Get current backend setting (opencv-rust specific)
 * @returns {string} Current backend
 */
export function getBackend() {
  return wasmBindings.getBackend();
}

/**
 * Check if GPU is available (opencv-rust specific)
 * @returns {Promise<boolean>} True if GPU available
 */
export async function isGpuAvailable() {
  await ensureInit();
  return await wasmBindings.isGpuAvailable();
}

// ============================================================================
// Image Filtering
// ============================================================================

/**
 * Gaussian blur filter
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image (modified in-place for compatibility)
 * @param {Object|Array} ksize - Kernel size {width, height} or [width, height]
 * @param {number} sigmaX - Gaussian kernel standard deviation in X direction
 * @param {number} sigmaY - Gaussian kernel standard deviation in Y direction (optional)
 * @param {number} borderType - Border mode (optional)
 */
export async function GaussianBlur(src, dst, ksize, sigmaX, sigmaY = 0, borderType = 0) {
  await ensureInit();
  const ksizeVal = Array.isArray(ksize) ? ksize[0] : ksize.width;
  const wasmResult = await wasmBindings.gaussianBlur(src._getWasmMat(), ksizeVal, sigmaX);
  dst._setWasmMat(wasmResult);
}

/**
 * Box blur filter
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {Object|Array} ksize - Kernel size
 * @param {Object} anchor - Anchor point (optional)
 * @param {boolean} normalize - Whether to normalize (optional)
 * @param {number} borderType - Border mode (optional)
 */
export async function blur(src, dst, ksize, anchor = {x: -1, y: -1}, normalize = true, borderType = 0) {
  await ensureInit();
  const ksizeVal = Array.isArray(ksize) ? ksize[0] : ksize.width;
  const wasmResult = await wasmBindings.blur(src._getWasmMat(), ksizeVal);
  dst._setWasmMat(wasmResult);
}

/**
 * Median blur filter
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} ksize - Kernel size (must be odd)
 */
export async function medianBlur(src, dst, ksize) {
  await ensureInit();
  const wasmResult = await wasmBindings.medianBlur(src._getWasmMat(), ksize);
  dst._setWasmMat(wasmResult);
}

/**
 * Bilateral filter (edge-preserving)
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} d - Diameter of pixel neighborhood
 * @param {number} sigmaColor - Filter sigma in color space
 * @param {number} sigmaSpace - Filter sigma in coordinate space
 * @param {number} borderType - Border mode (optional)
 */
export async function bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType = 0) {
  await ensureInit();
  const wasmResult = await wasmBindings.bilateralFilter(src._getWasmMat(), d, sigmaColor, sigmaSpace);
  dst._setWasmMat(wasmResult);
}

/**
 * Custom 2D filter
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} ddepth - Desired depth of destination image
 * @param {Mat} kernel - Convolution kernel
 * @param {Object} anchor - Anchor point (optional)
 * @param {number} delta - Value added to filtered pixels (optional)
 * @param {number} borderType - Border mode (optional)
 */
export async function filter2D(src, dst, ddepth, kernel, anchor = {x: -1, y: -1}, delta = 0, borderType = 0) {
  await ensureInit();
  // Convert kernel Mat to flat array
  const kernelData = kernel.data();
  const kernelArray = Array.from(kernelData).map(v => v / 255.0); // Normalize to 0-1
  const wasmResult = await wasmBindings.filter2D(
    src._getWasmMat(),
    kernelArray,
    kernel.cols(),
    kernel.rows()
  );
  dst._setWasmMat(wasmResult);
}

// ============================================================================
// Edge Detection
// ============================================================================

/**
 * Canny edge detection
 * @param {Mat} image - Source image
 * @param {Mat} edges - Output edge map
 * @param {number} threshold1 - First threshold for hysteresis
 * @param {number} threshold2 - Second threshold for hysteresis
 * @param {number} apertureSize - Aperture size for Sobel operator (optional)
 * @param {boolean} L2gradient - Use L2 norm for gradient (optional)
 */
export async function Canny(image, edges, threshold1, threshold2, apertureSize = 3, L2gradient = false) {
  await ensureInit();
  const wasmResult = await wasmBindings.canny(image._getWasmMat(), threshold1, threshold2);
  edges._setWasmMat(wasmResult);
}

/**
 * Sobel edge detection
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} ddepth - Output image depth
 * @param {number} dx - Order of derivative in x
 * @param {number} dy - Order of derivative in y
 * @param {number} ksize - Size of extended Sobel kernel (optional)
 * @param {number} scale - Scale factor (optional)
 * @param {number} delta - Delta value (optional)
 * @param {number} borderType - Border mode (optional)
 */
export async function Sobel(src, dst, ddepth, dx, dy, ksize = 3, scale = 1, delta = 0, borderType = 0) {
  await ensureInit();
  const wasmResult = await wasmBindings.sobel(src._getWasmMat(), dx, dy, ksize);
  dst._setWasmMat(wasmResult);
}

/**
 * Scharr edge detection
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} ddepth - Output image depth
 * @param {number} dx - Order of derivative in x
 * @param {number} dy - Order of derivative in y
 * @param {number} scale - Scale factor (optional)
 * @param {number} delta - Delta value (optional)
 * @param {number} borderType - Border mode (optional)
 */
export async function Scharr(src, dst, ddepth, dx, dy, scale = 1, delta = 0, borderType = 0) {
  await ensureInit();
  const wasmResult = await wasmBindings.scharr(src._getWasmMat(), dx, dy);
  dst._setWasmMat(wasmResult);
}

/**
 * Laplacian edge detection
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} ddepth - Output image depth
 * @param {number} ksize - Aperture size (optional)
 * @param {number} scale - Scale factor (optional)
 * @param {number} delta - Delta value (optional)
 * @param {number} borderType - Border mode (optional)
 */
export async function Laplacian(src, dst, ddepth, ksize = 1, scale = 1, delta = 0, borderType = 0) {
  await ensureInit();
  const wasmResult = await wasmBindings.laplacian(src._getWasmMat(), ksize);
  dst._setWasmMat(wasmResult);
}

// ============================================================================
// Thresholding
// ============================================================================

/**
 * Fixed-level threshold
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} thresh - Threshold value
 * @param {number} maxval - Maximum value for THRESH_BINARY
 * @param {number} type - Threshold type
 * @returns {Promise<number>} Computed threshold value
 */
export async function threshold(src, dst, thresh, maxval, type) {
  await ensureInit();
  const wasmResult = await wasmBindings.threshold(src._getWasmMat(), thresh, maxval, type);
  dst._setWasmMat(wasmResult);
  return thresh; // Return threshold used
}

/**
 * Adaptive threshold
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} maxValue - Maximum value
 * @param {number} adaptiveMethod - Adaptive method (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
 * @param {number} thresholdType - Threshold type (THRESH_BINARY or THRESH_BINARY_INV)
 * @param {number} blockSize - Size of pixel neighborhood
 * @param {number} C - Constant subtracted from mean
 */
export async function adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C) {
  await ensureInit();
  const wasmResult = await wasmBindings.adaptiveThreshold(
    src._getWasmMat(),
    maxValue,
    adaptiveMethod,
    thresholdType,
    blockSize,
    C
  );
  dst._setWasmMat(wasmResult);
}

// ============================================================================
// Morphological Operations
// ============================================================================

/**
 * Erode image
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {Mat} kernel - Structuring element
 * @param {Object} anchor - Anchor point (optional)
 * @param {number} iterations - Number of times erosion is applied (optional)
 * @param {number} borderType - Border mode (optional)
 * @param {Object} borderValue - Border value (optional)
 */
export async function erode(src, dst, kernel, anchor = {x: -1, y: -1}, iterations = 1, borderType = 0, borderValue = {}) {
  await ensureInit();
  const kernelSize = kernel ? kernel.cols() : 3;
  const wasmResult = await wasmBindings.erode(src._getWasmMat(), kernelSize, iterations);
  dst._setWasmMat(wasmResult);
}

/**
 * Dilate image
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {Mat} kernel - Structuring element
 * @param {Object} anchor - Anchor point (optional)
 * @param {number} iterations - Number of times dilation is applied (optional)
 * @param {number} borderType - Border mode (optional)
 * @param {Object} borderValue - Border value (optional)
 */
export async function dilate(src, dst, kernel, anchor = {x: -1, y: -1}, iterations = 1, borderType = 0, borderValue = {}) {
  await ensureInit();
  const kernelSize = kernel ? kernel.cols() : 3;
  const wasmResult = await wasmBindings.dilate(src._getWasmMat(), kernelSize, iterations);
  dst._setWasmMat(wasmResult);
}

/**
 * Advanced morphological transformations
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} op - Type of morphological operation
 * @param {Mat} kernel - Structuring element
 * @param {Object} anchor - Anchor point (optional)
 * @param {number} iterations - Number of iterations (optional)
 * @param {number} borderType - Border mode (optional)
 * @param {Object} borderValue - Border value (optional)
 */
export async function morphologyEx(src, dst, op, kernel, anchor = {x: -1, y: -1}, iterations = 1, borderType = 0, borderValue = {}) {
  await ensureInit();
  const kernelSize = kernel ? kernel.cols() : 3;
  const wasmResult = await wasmBindings.morphologyEx(src._getWasmMat(), op, kernelSize);
  dst._setWasmMat(wasmResult);
}

/**
 * Get structuring element for morphological operations
 * @param {number} shape - Element shape (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
 * @param {Object|Array} ksize - Size of structuring element
 * @param {Object} anchor - Anchor point (optional)
 * @returns {Mat} Structuring element
 */
export function getStructuringElement(shape, ksize, anchor = {x: -1, y: -1}) {
  const ksizeVal = Array.isArray(ksize) ? ksize[0] : ksize.width;
  const wasmResult = wasmBindings.getStructuringElement(shape, ksizeVal);
  const mat = new Mat();
  mat._setWasmMat(wasmResult);
  return mat;
}

// ============================================================================
// Color Space Conversions
// ============================================================================

/**
 * Convert between color spaces
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} code - Color conversion code
 * @param {number} dstCn - Number of channels in destination (optional)
 */
export async function cvtColor(src, dst, code, dstCn = 0) {
  await ensureInit();
  let wasmResult;

  switch (code) {
    case COLOR_RGB2GRAY:
    case COLOR_BGR2GRAY:
      wasmResult = await wasmBindings.cvtColorGray(src._getWasmMat());
      break;
    case COLOR_RGB2HSV:
    case COLOR_BGR2HSV:
      wasmResult = await wasmBindings.cvtColorHsv(src._getWasmMat());
      break;
    case COLOR_HSV2RGB:
    case COLOR_HSV2BGR:
      wasmResult = await wasmBindings.cvtColorHsvToRgb(src._getWasmMat());
      break;
    case COLOR_RGB2Lab:
    case COLOR_BGR2Lab:
      wasmResult = await wasmBindings.cvtColorLab(src._getWasmMat());
      break;
    case COLOR_Lab2RGB:
    case COLOR_Lab2BGR:
      wasmResult = await wasmBindings.cvtColorLabToRgb(src._getWasmMat());
      break;
    case COLOR_RGB2YCrCb:
    case COLOR_BGR2YCrCb:
      wasmResult = await wasmBindings.cvtColorYCrCb(src._getWasmMat());
      break;
    case COLOR_YCrCb2RGB:
    case COLOR_YCrCb2BGR:
      wasmResult = await wasmBindings.cvtColorYCrCbToRgb(src._getWasmMat());
      break;
    case COLOR_BGR2RGB:
    case COLOR_RGB2BGR:
      wasmResult = await wasmBindings.cvtColorBgr(src._getWasmMat());
      break;
    case COLOR_RGB2XYZ:
    case COLOR_BGR2XYZ:
      wasmResult = await wasmBindings.cvtColorXyz(src._getWasmMat());
      break;
    case COLOR_XYZ2RGB:
    case COLOR_XYZ2BGR:
      wasmResult = await wasmBindings.cvtColorXyzToRgb(src._getWasmMat());
      break;
    default:
      throw new Error(`Unsupported color conversion code: ${code}`);
  }

  dst._setWasmMat(wasmResult);
}

// ============================================================================
// Geometric Transformations
// ============================================================================

/**
 * Resize image
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {Object|Array} dsize - Output image size
 * @param {number} fx - Scale factor along horizontal axis (optional)
 * @param {number} fy - Scale factor along vertical axis (optional)
 * @param {number} interpolation - Interpolation method (optional)
 */
export async function resize(src, dst, dsize, fx = 0, fy = 0, interpolation = INTER_LINEAR) {
  await ensureInit();
  const width = Array.isArray(dsize) ? dsize[0] : dsize.width;
  const height = Array.isArray(dsize) ? dsize[1] : dsize.height;
  const wasmResult = await wasmBindings.resize(src._getWasmMat(), width, height, interpolation);
  dst._setWasmMat(wasmResult);
}

/**
 * Flip image
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} flipCode - Flip code (0=vertical, >0=horizontal, <0=both)
 */
export async function flip(src, dst, flipCode) {
  await ensureInit();
  const wasmResult = await wasmBindings.flip(src._getWasmMat(), flipCode);
  dst._setWasmMat(wasmResult);
}

/**
 * Rotate image
 * @param {Mat} src - Source image
 * @param {Mat} dst - Destination image
 * @param {number} rotateCode - Rotation code
 */
export async function rotate(src, dst, rotateCode) {
  await ensureInit();
  // Convert rotate code to angle
  let angle = 0;
  if (rotateCode === 0) angle = 90;  // ROTATE_90_CLOCKWISE
  else if (rotateCode === 1) angle = 180; // ROTATE_180
  else if (rotateCode === 2) angle = 270; // ROTATE_90_COUNTERCLOCKWISE

  const wasmResult = await wasmBindings.rotate(src._getWasmMat(), angle);
  dst._setWasmMat(wasmResult);
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

/**
 * Add two images
 * @param {Mat} src1 - First source image
 * @param {Mat} src2 - Second source image
 * @param {Mat} dst - Destination image
 * @param {Mat} mask - Optional operation mask (optional)
 * @param {number} dtype - Output depth (optional)
 */
export async function add(src1, src2, dst, mask = null, dtype = -1) {
  await ensureInit();
  const wasmResult = await wasmBindings.add(src1._getWasmMat(), src2._getWasmMat());
  dst._setWasmMat(wasmResult);
}

/**
 * Subtract two images
 * @param {Mat} src1 - First source image
 * @param {Mat} src2 - Second source image
 * @param {Mat} dst - Destination image
 * @param {Mat} mask - Optional operation mask (optional)
 * @param {number} dtype - Output depth (optional)
 */
export async function subtract(src1, src2, dst, mask = null, dtype = -1) {
  await ensureInit();
  const wasmResult = await wasmBindings.subtract(src1._getWasmMat(), src2._getWasmMat());
  dst._setWasmMat(wasmResult);
}

/**
 * Multiply two images
 * @param {Mat} src1 - First source image
 * @param {Mat} src2 - Second source image
 * @param {Mat} dst - Destination image
 * @param {number} scale - Scale factor (optional)
 * @param {number} dtype - Output depth (optional)
 */
export async function multiply(src1, src2, dst, scale = 1, dtype = -1) {
  await ensureInit();
  const wasmResult = await wasmBindings.multiply(src1._getWasmMat(), src2._getWasmMat());
  dst._setWasmMat(wasmResult);
}

/**
 * Weighted sum of two images
 * @param {Mat} src1 - First source image
 * @param {number} alpha - Weight of first image
 * @param {Mat} src2 - Second source image
 * @param {number} beta - Weight of second image
 * @param {number} gamma - Scalar added to each sum
 * @param {Mat} dst - Destination image
 * @param {number} dtype - Output depth (optional)
 */
export async function addWeighted(src1, alpha, src2, beta, gamma, dst, dtype = -1) {
  await ensureInit();
  const wasmResult = await wasmBindings.addWeighted(
    src1._getWasmMat(),
    alpha,
    src2._getWasmMat(),
    beta,
    gamma
  );
  dst._setWasmMat(wasmResult);
}

// ============================================================================
// Histogram Operations
// ============================================================================

/**
 * Equalize histogram
 * @param {Mat} src - Source grayscale image
 * @param {Mat} dst - Destination image
 */
export async function equalizeHist(src, dst) {
  await ensureInit();
  const wasmResult = await wasmBindings.equalizeHistogram(src._getWasmMat());
  dst._setWasmMat(wasmResult);
}

// ============================================================================
// Utility & Information
// ============================================================================

/**
 * Get OpenCV version string
 * @returns {string} Version string
 */
export function getBuildInformation() {
  return `opencv-rust WASM ${wasmBindings.getVersion()}\nGPU: WebGPU Backend Available`;
}

/**
 * Get runtime info
 * @returns {Object} Runtime information
 */
export async function getRuntimeInfo() {
  await ensureInit();
  const gpuAvailable = await isGpuAvailable();
  return {
    version: wasmBindings.getVersion(),
    backend: getBackend(),
    gpuAvailable: gpuAvailable
  };
}

// ============================================================================
// Export default cv object (opencv.js style)
// ============================================================================

const cv = {
  // Core
  Mat,

  // Constants
  THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV,
  THRESH_OTSU, THRESH_TRIANGLE,
  ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C,
  COLOR_BGR2GRAY, COLOR_RGB2GRAY, COLOR_BGR2HSV, COLOR_RGB2HSV,
  COLOR_HSV2BGR, COLOR_HSV2RGB, COLOR_BGR2Lab, COLOR_RGB2Lab,
  COLOR_Lab2BGR, COLOR_Lab2RGB, COLOR_BGR2YCrCb, COLOR_RGB2YCrCb,
  COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, COLOR_BGR2RGB, COLOR_RGB2BGR,
  COLOR_BGR2XYZ, COLOR_RGB2XYZ, COLOR_XYZ2BGR, COLOR_XYZ2RGB,
  MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE,
  MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT,
  MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE,
  INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4,
  FLIP_HORIZONTAL, FLIP_VERTICAL, FLIP_BOTH,

  // Backend (opencv-rust specific)
  initGpu,
  setBackend,
  getBackend,
  isGpuAvailable,

  // Filtering
  GaussianBlur,
  blur,
  medianBlur,
  bilateralFilter,
  filter2D,

  // Edge Detection
  Canny,
  Sobel,
  Scharr,
  Laplacian,

  // Thresholding
  threshold,
  adaptiveThreshold,

  // Morphology
  erode,
  dilate,
  morphologyEx,
  getStructuringElement,

  // Color
  cvtColor,

  // Geometric
  resize,
  flip,
  rotate,

  // Arithmetic
  add,
  subtract,
  multiply,
  addWeighted,

  // Histogram
  equalizeHist,

  // Utility
  getBuildInformation,
  getRuntimeInfo,
};

export default cv;
