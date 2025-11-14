/**
 * Check which operations in demoRegistry.js are actually implemented in WASM
 */
import { readFileSync } from 'fs';

// Read demoRegistry.js and extract all demos marked as implemented
const demoRegistryContent = readFileSync('../examples/web-benchmark/src/demos/demoRegistry.js', 'utf-8');

// Extract demo objects
const demoMatches = demoRegistryContent.matchAll(/{\s*id:\s*'([^']+)'[^}]*implemented:\s*true/g);
const implementedDemos = Array.from(demoMatches, m => m[1]);

console.log(`Found ${implementedDemos.length} operations marked as implemented:\n`);

// Read list of exported WASM functions
const wasmExports = readFileSync('wasm_exports.txt', 'utf-8').trim().split('\n');

// Map demo IDs to expected WASM function names
const demoToWasmFunction = {
    // Filters
    'gaussian_blur': 'gaussianBlur',
    'box_blur': 'boxBlur',
    'median_blur': 'medianBlur',
    'bilateral_filter': 'bilateralFilter',
    'guided_filter': 'guidedFilter',
    'gabor_filter': 'gaborFilter',
    'log_filter': 'logFilter',
    'nlm_denoising': 'nlmDenoising',
    'anisotropic_diffusion': 'anisotropicDiffusion',
    'distance_transform': 'distanceTransform',
    'watershed': 'watershed',

    // Edges
    'canny': 'canny',
    'sobel': 'sobel',
    'scharr': 'scharr',
    'laplacian': 'laplacian',

    // Transform
    'resize': 'resize',
    'flip': 'flip',
    'rotate': 'rotate',
    'warp_affine': 'warpAffine',
    'warp_perspective': 'warpPerspective',
    'get_rotation_matrix_2d': 'getRotationMatrix2D',

    // Color
    'cvt_color_gray': 'cvtColorGray',
    'cvt_color_hsv': 'cvtColorHsv',
    'cvt_color_lab': 'cvtColorLab',
    'cvt_color_ycrcb': 'cvtColorYCrCb',
    'threshold': 'threshold',
    'adaptive_threshold': 'adaptiveThreshold',

    // Histogram
    'calc_histogram': 'calcHistogram',
    'equalize_histogram': 'equalizeHistogram',
    'normalize_histogram': 'normalizeHistogram',
    'compare_histograms': 'compareHistograms',
    'back_projection': 'backProjection',

    // Morphology
    'erode': 'erode',
    'dilate': 'dilate',
    'morphology_opening': 'morphologyOpening',
    'morphology_closing': 'morphologyClosing',
    'morphology_gradient': 'morphologyGradient',
    'morphology_tophat': 'morphologyTopHat',
    'morphology_blackhat': 'morphologyBlackHat',

    // Contours
    'find_contours': 'findContours',
    'approx_poly_dp': 'approxPolyDP',
    'contour_area': 'contourArea',
    'arc_length': 'arcLength',
    'bounding_rect': 'boundingRect',
    'moments': 'moments',

    // Features
    'harris_corners': 'harrisCorners',
    'good_features_to_track': 'goodFeaturesToTrack',
    'fast': 'fast',
    'sift': 'sift',
    'orb': 'orb',
    'brisk': 'brisk',
    'akaze': 'akaze',
    'kaze': 'kaze',
    'brute_force_matcher': 'bruteForceMatcher',

    // Hough
    'hough_lines': 'houghLines',
    'hough_lines_p': 'houghLinesP',
    'hough_circles': 'houghCircles',

    // Detection
    'hog_descriptor': 'hogDescriptor',
    'cascade_classifier': 'cascadeClassifier',
    'aruco_detector': 'detectAruco',
    'qr_detector': 'detectQR',

    // Video
    'farneback_optical_flow': 'farnebackOpticalFlow',
    'meanshift_tracker': 'meanshiftTracker',
    'camshift_tracker': 'camshiftTracker',
    'mosse_tracker': 'mosseTracker',
    'csrt_tracker': 'csrtTracker',
    'bg_subtractor_mog2': 'bgSubtractorMog2',
    'bg_subtractor_knn': 'bgSubtractorKnn',

    // Calibration
    'calibrate_camera': 'calibrateCamera',
    'fisheye_calibration': 'fisheyeCalibration',
    'solve_pnp': 'solvePnp',
    'stereo_calibration': 'stereoCalibration',
    'stereo_rectification': 'stereoRectification',
    'compute_disparity': 'computeDisparity',
    'find_homography': 'findHomography',

    // ML
    'svm_classifier': 'svmClassifier',
    'decision_tree': 'decisionTree',
    'random_forest': 'randomForest',
    'knn': 'knn',
    'neural_network': 'neuralNetwork',
    'kmeans': 'kmeans',

    // Photo
    'merge_debevec': 'mergeDebevec',
    'tonemap_drago': 'tonemapDrago',
    'tonemap_reinhard': 'tonemapReinhard',
    'fast_nl_means': 'fastNlMeans',
    'inpaint': 'inpaint',
    'super_resolution': 'superResolution',

    // Stitching
    'panorama_stitcher': 'panoramaStitcher',
    'feather_blender': 'featherBlender',
    'multiband_blender': 'multibandBlender',

    // Drawing
    'draw_line': 'drawLine',
    'draw_rectangle': 'drawRectangle',
    'draw_circle': 'drawCircle',
    'draw_ellipse': 'drawEllipse',
    'draw_polylines': 'drawPolylines',
    'put_text': 'putText',

    // DNN
    'load_network': 'loadNetwork',
    'blob_from_image': 'blobFromImage',

    // Shape
    'min_enclosing_circle': 'minEnclosingCircle',
    'convex_hull': 'convexHull',
    'hu_moments': 'huMoments',
    'match_shapes': 'matchShapes',
};

const missing = [];
const implemented = [];

for (const demoId of implementedDemos) {
    const wasmFn = demoToWasmFunction[demoId];

    if (!wasmFn) {
        console.log(`⚠️  ${demoId.padEnd(35)} - NO MAPPING DEFINED`);
        missing.push({ demo: demoId, reason: 'no mapping' });
        continue;
    }

    if (wasmExports.includes(wasmFn)) {
        implemented.push(demoId);
    } else {
        console.log(`✗  ${demoId.padEnd(35)} → ${wasmFn.padEnd(30)} MISSING`);
        missing.push({ demo: demoId, wasm: wasmFn, reason: 'not exported' });
    }
}

console.log(`\n${'='.repeat(80)}`);
console.log(`\nSummary:`);
console.log(`  ✓ Implemented: ${implemented.length}`);
console.log(`  ✗ Missing: ${missing.length}`);

if (missing.length > 0) {
    console.log(`\nMissing operations that need implementation:`);
    missing.forEach(({ demo, wasm, reason }) => {
        if (reason === 'not exported') {
            console.log(`  - ${demo} (${wasm})`);
        }
    });
}
