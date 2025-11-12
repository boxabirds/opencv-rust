/**
 * Pixel-level comparison utilities for OpenCV.js parity testing
 *
 * Compares two images pixel-by-pixel and calculates:
 * - Maximum pixel difference
 * - Mean pixel difference
 * - Outlier percentage (pixels exceeding tolerance)
 *
 * These metrics determine if our WASM implementation produces
 * sufficiently similar results to opencv.js reference.
 */

/**
 * Compare two images pixel-by-pixel
 *
 * @param {Object} imageA - First image { data: Uint8Array/Array, width, height }
 * @param {Object} imageB - Second image { data: Uint8Array/Array, width, height }
 * @param {Object} tolerance - Tolerance thresholds
 * @param {number} tolerance.max_pixel_diff - Maximum allowed pixel difference
 * @param {number} tolerance.max_mean_diff - Maximum allowed mean difference
 * @param {number} tolerance.max_outliers_percent - Maximum percentage of outliers
 * @returns {Object} Comparison results with metrics and pass/fail status
 */
export function compareImages(imageA, imageB, tolerance) {
  const dataA = imageA.data;
  const dataB = imageB.data;

  // Validate inputs
  if (!dataA || !dataB) {
    throw new Error('Image data is required');
  }

  if (dataA.length !== dataB.length) {
    throw new Error(
      `Image dimensions mismatch: ${dataA.length} vs ${dataB.length}`
    );
  }

  let maxDiff = 0;
  let totalDiff = 0;
  let outlierCount = 0;
  let diffCount = 0;

  // Compare pixel-by-pixel
  for (let i = 0; i < dataA.length; i++) {
    const diff = Math.abs(dataA[i] - dataB[i]);

    if (diff > 0) {
      diffCount++;
      totalDiff += diff;
      maxDiff = Math.max(maxDiff, diff);

      // Count outliers (pixels exceeding base tolerance)
      if (diff > tolerance.max_pixel_diff) {
        outlierCount++;
      }
    }
  }

  const meanDiff = diffCount > 0 ? totalDiff / diffCount : 0;
  const outlierPercent = (outlierCount / dataA.length) * 100;

  // Determine pass/fail
  // Allow 2x max_pixel_diff for occasional outliers, but enforce mean and outlier %
  const passed =
    maxDiff <= tolerance.max_pixel_diff * 2 &&
    meanDiff <= tolerance.max_mean_diff &&
    outlierPercent <= tolerance.max_outliers_percent;

  return {
    passed,
    maxDiff,
    meanDiff,
    outlierPercent,
    totalPixels: dataA.length,
    differentPixels: diffCount,
    outlierPixels: outlierCount,
    tolerance,
  };
}

/**
 * Format comparison results for human-readable output
 *
 * @param {Object} comparison - Results from compareImages()
 * @returns {string} Formatted comparison report
 */
export function formatComparisonReport(comparison) {
  const status = comparison.passed ? '✅ PASS' : '❌ FAIL';

  return `
${status}

Pixels Compared: ${comparison.totalPixels}
Different Pixels: ${comparison.differentPixels} (${(
    (comparison.differentPixels / comparison.totalPixels) *
    100
  ).toFixed(2)}%)

Max Difference: ${comparison.maxDiff.toFixed(2)} (threshold: ${
    comparison.tolerance.max_pixel_diff * 2
  })
Mean Difference: ${comparison.meanDiff.toFixed(4)} (threshold: ${
    comparison.tolerance.max_mean_diff
  })
Outliers: ${comparison.outlierPercent.toFixed(2)}% (threshold: ${
    comparison.tolerance.max_outliers_percent
  }%)

Outlier Pixels: ${comparison.outlierPixels} (exceeding ${
    comparison.tolerance.max_pixel_diff
  }px diff)
`.trim();
}

/**
 * Compare images with detailed per-channel breakdown
 *
 * Useful for debugging color space issues or channel-specific problems
 *
 * @param {Object} imageA - First image
 * @param {Object} imageB - Second image
 * @param {Object} tolerance - Tolerance thresholds
 * @param {number} channels - Number of channels (3 for RGB, 4 for RGBA)
 * @returns {Object} Comparison with per-channel metrics
 */
export function compareImagesDetailed(imageA, imageB, tolerance, channels = 4) {
  const dataA = imageA.data;
  const dataB = imageB.data;

  const channelNames = ['R', 'G', 'B', 'A'];
  const channelStats = [];

  // Per-channel comparison
  for (let c = 0; c < channels; c++) {
    let maxDiff = 0;
    let totalDiff = 0;
    let diffCount = 0;

    for (let i = c; i < dataA.length; i += channels) {
      const diff = Math.abs(dataA[i] - dataB[i]);
      if (diff > 0) {
        diffCount++;
        totalDiff += diff;
        maxDiff = Math.max(maxDiff, diff);
      }
    }

    const meanDiff = diffCount > 0 ? totalDiff / diffCount : 0;

    channelStats.push({
      channel: channelNames[c],
      maxDiff,
      meanDiff,
      differentPixels: diffCount,
    });
  }

  // Overall comparison
  const overall = compareImages(imageA, imageB, tolerance);

  return {
    ...overall,
    channelStats,
  };
}

/**
 * Generate a difference map showing where images differ
 *
 * Creates a new image where brightness indicates difference magnitude
 *
 * @param {Object} imageA - First image
 * @param {Object} imageB - Second image
 * @returns {Uint8Array} Difference map (same size as input)
 */
export function generateDifferenceMap(imageA, imageB) {
  const dataA = imageA.data;
  const dataB = imageB.data;
  const diffMap = new Uint8Array(dataA.length);

  for (let i = 0; i < dataA.length; i += 4) {
    // Calculate per-pixel difference (average across RGB)
    const diffR = Math.abs(dataA[i] - dataB[i]);
    const diffG = Math.abs(dataA[i + 1] - dataB[i + 1]);
    const diffB = Math.abs(dataA[i + 2] - dataB[i + 2]);

    const avgDiff = (diffR + diffG + diffB) / 3;

    // Scale difference for visibility (multiply by 10)
    const scaled = Math.min(255, avgDiff * 10);

    // Set to grayscale
    diffMap[i] = scaled; // R
    diffMap[i + 1] = scaled; // G
    diffMap[i + 2] = scaled; // B
    diffMap[i + 3] = 255; // A
  }

  return diffMap;
}
