import { useEffect, useRef } from 'react';
import init, {
  WasmMat,
  initGpu,
  isGpuAvailable,
  gaussianBlur as wasmGaussianBlur,
  resize as wasmResize,
  threshold as wasmThreshold,
  canny as wasmCanny,
  blur as wasmBlur,
  medianBlur as wasmMedianBlur,
  bilateralFilter as wasmBilateralFilter,
  sobel as wasmSobel,
  scharr as wasmScharr,
  laplacian as wasmLaplacian,
  flip as wasmFlip,
  rotate as wasmRotate,
  cvtColorGray as wasmCvtColorGray,
  adaptiveThreshold as wasmAdaptiveThreshold,
  drawLine as wasmDrawLine,
  drawRectangle as wasmDrawRectangle,
  drawCircle as wasmDrawCircle,
  guidedFilter as wasmGuidedFilter,
  gaborFilter as wasmGaborFilter,
  warpAffine as wasmWarpAffine,
  harrisCorners as wasmHarrisCorners,
  goodFeaturesToTrack as wasmGoodFeaturesToTrack,
  fast as wasmFast,
  erode as wasmErode,
  dilate as wasmDilate,
  morphologyOpening as wasmMorphologyOpening,
  morphologyClosing as wasmMorphologyClosing,
  morphologyGradient as wasmMorphologyGradient,
  equalizeHistogram as wasmEqualizeHistogram,
  cvtColorHsv as wasmCvtColorHsv,
  distanceTransform as wasmDistanceTransform,
  nlmDenoising as wasmNlmDenoising,
  houghLines as wasmHoughLines,
  houghLinesP as wasmHoughLinesP,
  houghCircles as wasmHoughCircles,
  findContours as wasmFindContours,
  boundingRect as wasmBoundingRect,
  calcHistogram as wasmCalcHistogram,
  detectAruco as wasmDetectAruco,
  detectQR as wasmDetectQR,
  contourArea as wasmContourArea,
  arcLength as wasmArcLength,
  approxPolyDP as wasmApproxPolyDP,
  anisotropicDiffusion as wasmAnisotropicDiffusion,
  morphologyTophat as wasmMorphologyTophat,
  morphologyBlackhat as wasmMorphologyBlackhat,
  warpPerspective as wasmWarpPerspective,
  getRotationMatrix2D as wasmGetRotationMatrix2d,
  normalizeHistogram as wasmNormalizeHistogram,
  compareHistograms as wasmCompareHistograms,
  backProjection as wasmBackProjection,
  moments as wasmMoments,
  watershed as wasmWatershed,
  sift as wasmSift,
  orb as wasmOrb,
  brisk as wasmBrisk,
  akaze as wasmAkaze,
  kaze as wasmKaze,
  logFilter as wasmLogFilter,
  cvtColorLab as wasmCvtColorLab,
  cvtColorYCrCb as wasmCvtColorYCrCb,
  drawEllipse as wasmDrawEllipse,
  drawPolylines as wasmDrawPolylines,
  putText as wasmPutText,
  minEnclosingCircle as wasmMinEnclosingCircle,
  convexHull as wasmConvexHull,
  huMoments as wasmHuMoments,
  matchShapes as wasmMatchShapes,
  inpaint as wasmInpaint,
  kmeans as wasmKmeans,
  findHomography as wasmFindHomography,
  bruteForceMatcher as wasmBruteForceMatcher,
  hogDescriptor as wasmHogDescriptor,
  bgSubtractorMog2 as wasmBgSubtractorMog2,
  bgSubtractorKnn as wasmBgSubtractorKnn,
  farnebackOpticalFlow as wasmFarnebackOpticalFlow,
  tonemapDrago as wasmTonemapDrago,
  tonemapReinhard as wasmTonemapReinhard,
  meanshiftTracker as wasmMeanshiftTracker,
  camshiftTracker as wasmCamshiftTracker,
  mosseTracker as wasmMosseTracker,
  csrtTracker as wasmCsrtTracker,
  svmClassifier as wasmSvmClassifier,
  decisionTree as wasmDecisionTree,
  randomForest as wasmRandomForest,
  knn as wasmKnn,
  neuralNetwork as wasmNeuralNetwork,
  fastNlMeans as wasmFastNlMeans,
  superResolution as wasmSuperResolution,
  mergeDebevec as wasmMergeDebevec,
  calibrateCamera as wasmCalibrateCamera,
  fisheyeCalibration as wasmFisheyeCalibration,
  solvePnp as wasmSolvePnp,
  stereoCalibration as wasmStereoCalibration,
  computeDisparity as wasmComputeDisparity,
  cascadeClassifier as wasmCascadeClassifier,
  panoramaStitcher as wasmPanoramaStitcher,
  featherBlender as wasmFeatherBlender,
  stereoRectification as wasmStereoRectification,
  multibandBlender as wasmMultibandBlender,
  loadNetwork as wasmLoadNetwork,
  blobFromImage as wasmBlobFromImage,
  getVersion
} from '../../../pkg/opencv_rust.js';

import { useAppStore } from './store/appStore';
import { imageToImageData, matToImageDataURL, createThumbnail } from './utils/imageUtils';
import { getDemoById } from './demos/demoRegistry';

import Sidebar from './components/Sidebar';
import DemoControls from './components/DemoControls';
import InputOutput from './components/InputOutput';
import PerformanceMetrics from './components/PerformanceMetrics';
import History from './components/History';

import './App.css';

function App() {
  const {
    wasmLoaded,
    gpuAvailable,
    setWasmLoaded,
    setGpuAvailable,
    selectedDemo,
    demoParams,
    inputImage,
    setOutputImage,
    setProcessing,
    setPerformance,
    addToHistory
  } = useAppStore();

  // Initialize WASM and WebGPU once
  const initializedRef = useRef(false);

  useEffect(() => {
    const initWasm = async () => {
      if (initializedRef.current) return;
      initializedRef.current = true;

      try {
        console.log('Initializing WASM module...');
        await init();

        console.log('Initializing WebGPU...');
        const gpuInit = await initGpu();
        setGpuAvailable(gpuInit);

        if (gpuInit) {
          console.log('✓ WebGPU initialized successfully');
        } else {
          console.warn('WebGPU initialization failed - falling back to CPU');
        }

        const version = getVersion();
        console.log(`OpenCV-Rust WASM loaded! Version: ${version}`);
        setWasmLoaded(true);
      } catch (error) {
        console.error('Failed to initialize WASM:', error);
        setWasmLoaded(false);
      }
    };

    initWasm();
  }, []);

  const processImage = async () => {
    if (!inputImage || !wasmLoaded) {
      alert('Please upload an image first');
      return;
    }

    const demo = getDemoById(selectedDemo);
    if (!demo || !demo.implemented) {
      alert('This demo is not yet implemented');
      return;
    }

    setProcessing(true);
    let cpuTime = null;
    let gpuTime = null;
    let resultImage = null;

    try {
      // Load image to ImageData
      const imageData = await imageToImageData(inputImage.dataURL);
      console.log(`[${demo.id}] Input ImageData pixels:`, {
        firstPixels: Array.from(imageData.data.slice(0, 16)),
        someMiddle: Array.from(imageData.data.slice(1000, 1016))
      });

      // Create WASM Mat
      const srcMat = WasmMat.fromImageData(
        imageData.data,
        imageData.width,
        imageData.height,
        4
      );

      // Process with GPU (if available)
      if (gpuAvailable) {
        console.log(`[${demo.id}] Starting GPU processing...`);
        const startGpu = performance.now();
        const gpuResult = await runDemo(demo.id, srcMat, demoParams);
        const endGpu = performance.now();
        gpuTime = endGpu - startGpu;
        console.log(`[${demo.id}] GPU time: ${gpuTime}ms, result:`, gpuResult);

        if (gpuResult) {
          const resultData = gpuResult.getData();
          console.log(`[${demo.id}] Converting to image URL...`, {
            width: gpuResult.width,
            height: gpuResult.height,
            channels: gpuResult.channels,
            dataLength: resultData.length,
            firstPixels: Array.from(resultData.slice(0, 16)),
            someMiddlePixels: Array.from(resultData.slice(1000, 1016))
          });
          resultImage = matToImageDataURL(gpuResult);
          console.log(`[${demo.id}] Result image URL:`, resultImage ? `data:image/... (${resultImage.length} chars)` : 'NULL');
          if (resultImage) {
            console.log(`[${demo.id}] Data URL preview:`, resultImage.substring(0, 100));
          }
          gpuResult.free();
        } else {
          console.warn(`[${demo.id}] GPU result is null/undefined`);
        }
      }

      // Process with CPU for comparison
      const startCpu = performance.now();
      const cpuResult = await runDemo(demo.id, srcMat, demoParams);
      const endCpu = performance.now();
      cpuTime = endCpu - startCpu;

      if (cpuResult && !resultImage) {
        resultImage = matToImageDataURL(cpuResult);
      }
      if (cpuResult) {
        cpuResult.free();
      }

      // Clean up source mat
      srcMat.free();

      // Update UI
      console.log(`[${demo.id}] About to update UI, resultImage:`, resultImage ? 'exists' : 'NULL');
      if (resultImage) {
        console.log(`[${demo.id}] Calling setOutputImage...`);
        setOutputImage(resultImage);
        setPerformance(cpuTime, gpuTime);
        console.log(`[${demo.id}] UI updated`);

        // Add to history
        const thumbnail = await createThumbnail(resultImage);
        addToHistory({
          category: demo.category,
          demo: demo.id,
          params: { ...demoParams },
          inputImage: inputImage.dataURL,
          outputImage: resultImage,
          outputThumbnail: thumbnail,
          processingTime: gpuTime || cpuTime
        });
      } else {
        console.error(`[${demo.id}] No result image to display!`);
      }
    } catch (error) {
      console.error('Processing failed:', error);
      alert(`Processing failed: ${error.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const runDemo = async (demoId, srcMat, params) => {
    switch (demoId) {
      case 'gaussian_blur': {
        const ksize = params.ksize || 5;
        const sigma = params.sigma || 1.5;
        return await wasmGaussianBlur(srcMat, ksize, sigma);
      }

      case 'resize': {
        const scale = params.scale || 0.5;
        const newWidth = Math.floor(srcMat.width * scale);
        const newHeight = Math.floor(srcMat.height * scale);
        return await wasmResize(srcMat, newWidth, newHeight);
      }

      case 'threshold': {
        const thresh = params.thresh || 127;
        const maxval = params.maxval || 255;
        return await wasmThreshold(srcMat, thresh, maxval);
      }

      case 'canny': {
        const threshold1 = params.threshold1 || 50;
        const threshold2 = params.threshold2 || 150;
        return await wasmCanny(srcMat, threshold1, threshold2);
      }

      case 'box_blur': {
        const ksize = params.ksize || 5;
        return await wasmBlur(srcMat, ksize);
      }

      case 'median_blur': {
        const ksize = params.ksize || 5;
        return await wasmMedianBlur(srcMat, ksize);
      }

      case 'bilateral_filter': {
        const d = params.diameter || 9;
        const sigmaColor = params.sigmaColor || 75;
        const sigmaSpace = params.sigmaSpace || 75;
        return await wasmBilateralFilter(srcMat, d, sigmaColor, sigmaSpace);
      }

      case 'sobel': {
        const dx = params.dx || 1;
        const dy = params.dy || 0;
        const ksize = params.ksize || 3;
        return await wasmSobel(srcMat, dx, dy, ksize);
      }

      case 'scharr': {
        const dx = params.dx || 1;
        const dy = params.dy || 0;
        return await wasmScharr(srcMat, dx, dy);
      }

      case 'laplacian': {
        const ksize = params.ksize || 3;
        return await wasmLaplacian(srcMat, ksize);
      }

      case 'flip': {
        // Convert 'Horizontal', 'Vertical', 'Both' to flip codes
        let flipCode = 1; // Default: horizontal
        if (params.flipCode === 'Vertical') flipCode = 0;
        else if (params.flipCode === 'Both') flipCode = -1;
        else if (typeof params.flipCode === 'number') flipCode = params.flipCode;
        return await wasmFlip(srcMat, flipCode);
      }

      case 'rotate': {
        // Convert '90°', '180°', '270°' to rotation codes
        let rotateCode = 0; // 90° clockwise
        if (params.angle === '180°' || params.angle === 180) rotateCode = 1;
        else if (params.angle === '270°' || params.angle === 270) rotateCode = 2;
        else if (typeof params.rotateCode === 'number') rotateCode = params.rotateCode;
        return await wasmRotate(srcMat, rotateCode);
      }

      case 'cvt_color_gray': {
        return await wasmCvtColorGray(srcMat);
      }

      case 'adaptive_threshold': {
        const maxval = params.maxval || 255;
        const blockSize = params.blockSize || 11;
        const c = params.c || 2;
        return await wasmAdaptiveThreshold(srcMat, maxval, blockSize, c);
      }

      case 'draw_line': {
        // Use center of image for line endpoints as demo
        const width = srcMat.width;
        const height = srcMat.height;
        const x1 = Math.floor(width * 0.2);
        const y1 = Math.floor(height * 0.2);
        const x2 = Math.floor(width * 0.8);
        const y2 = Math.floor(height * 0.8);
        const thickness = params.thickness || 2;
        // Parse color - default to red
        const color = params.color || '#FF0000';
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        return await wasmDrawLine(srcMat, x1, y1, x2, y2, r, g, b, thickness);
      }

      case 'draw_rectangle': {
        // Draw rectangle in center third of image as demo
        const width = srcMat.width;
        const height = srcMat.height;
        const x = Math.floor(width * 0.25);
        const y = Math.floor(height * 0.25);
        const w = Math.floor(width * 0.5);
        const h = Math.floor(height * 0.5);
        const thickness = params.thickness || 2;
        const color = params.color || '#00FF00';
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        return await wasmDrawRectangle(srcMat, x, y, w, h, r, g, b, thickness);
      }

      case 'draw_circle': {
        // Draw circle in center of image as demo
        const centerX = Math.floor(srcMat.width / 2);
        const centerY = Math.floor(srcMat.height / 2);
        const radius = params.radius || Math.floor(Math.min(srcMat.width, srcMat.height) * 0.2);
        const color = params.color || '#0000FF';
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        return await wasmDrawCircle(srcMat, centerX, centerY, radius, r, g, b);
      }

      case 'guided_filter': {
        const radius = params.radius || 5;
        const eps = params.epsilon || 0.1;
        return await wasmGuidedFilter(srcMat, radius, eps);
      }

      case 'gabor_filter': {
        const ksize = 21;
        const sigma = params.sigma || 3.0;
        const theta = (params.orientation || 0) * Math.PI / 180; // Convert degrees to radians
        const lambda = 1.0 / (params.frequency || 0.1);
        const gamma = 0.5;
        const psi = 0;
        return await wasmGaborFilter(srcMat, ksize, sigma, theta, lambda, gamma, psi);
      }

      case 'warp_affine': {
        // Create a simple rotation + translation matrix as demo
        const angle = (params.angle || 15) * Math.PI / 180;
        const scale = params.scale || 1.0;
        const centerX = srcMat.width / 2;
        const centerY = srcMat.height / 2;

        // Rotation matrix around center
        const cos = Math.cos(angle) * scale;
        const sin = Math.sin(angle) * scale;
        const tx = centerX - centerX * cos + centerY * sin;
        const ty = centerY - centerX * sin - centerY * cos;

        const matrix = [cos, -sin, tx, sin, cos, ty];
        return await wasmWarpAffine(srcMat, matrix, srcMat.width, srcMat.height);
      }

      case 'harris_corners': {
        const blockSize = params.blockSize || 3;
        const ksize = params.ksize || 3;
        const k = params.k || 0.04;
        const threshold = params.threshold || 100.0;
        return await wasmHarrisCorners(srcMat, blockSize, ksize, k, threshold);
      }

      case 'good_features_to_track': {
        const maxCorners = params.maxCorners || 100;
        const qualityLevel = params.qualityLevel || 0.01;
        const minDistance = params.minDistance || 10.0;
        const blockSize = params.blockSize || 3;
        return await wasmGoodFeaturesToTrack(srcMat, maxCorners, qualityLevel, minDistance, blockSize);
      }

      case 'fast': {
        const threshold = params.threshold || 20;
        const nonmaxSuppression = params.nonmaxSuppression !== false;
        return await wasmFast(srcMat, threshold, nonmaxSuppression);
      }

      case 'erode': {
        const ksize = params.ksize || 5;
        return await wasmErode(srcMat, ksize);
      }

      case 'dilate': {
        const ksize = params.ksize || 5;
        return await wasmDilate(srcMat, ksize);
      }

      case 'morphology_opening': {
        const ksize = params.ksize || 5;
        return await wasmMorphologyOpening(srcMat, ksize);
      }

      case 'morphology_closing': {
        const ksize = params.ksize || 5;
        return await wasmMorphologyClosing(srcMat, ksize);
      }

      case 'morphology_gradient': {
        const ksize = params.ksize || 5;
        return await wasmMorphologyGradient(srcMat, ksize);
      }

      case 'equalize_histogram': {
        return await wasmEqualizeHistogram(srcMat);
      }

      case 'cvt_color_hsv': {
        return await wasmCvtColorHsv(srcMat);
      }

      case 'distance_transform': {
        return await wasmDistanceTransform(srcMat);
      }

      case 'nlm_denoising': {
        const h = params.h || 10.0;
        const templateWindowSize = params.templateWindowSize || 7;
        const searchWindowSize = params.searchWindowSize || 21;
        return await wasmNlmDenoising(srcMat, h, templateWindowSize, searchWindowSize);
      }

      case 'hough_lines': {
        const threshold = params.threshold || 100;
        return await wasmHoughLines(srcMat, threshold);
      }

      case 'hough_lines_p': {
        const threshold = params.threshold || 50;
        const minLineLength = params.minLineLength || 50.0;
        const maxLineGap = params.maxLineGap || 10.0;
        return await wasmHoughLinesP(srcMat, threshold, minLineLength, maxLineGap);
      }

      case 'hough_circles': {
        const minDist = params.minDist || 50.0;
        const param1 = params.param1 || 100.0;
        const param2 = params.param2 || 30.0;
        const minRadius = params.minRadius || 10;
        const maxRadius = params.maxRadius || 100;
        return await wasmHoughCircles(srcMat, minDist, param1, param2, minRadius, maxRadius);
      }

      case 'find_contours': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmFindContours(srcMat, thresholdValue);
      }

      case 'bounding_rect': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmBoundingRect(srcMat, thresholdValue);
      }

      case 'calc_histogram': {
        return await wasmCalcHistogram(srcMat);
      }

      case 'aruco_detector': {
        const dictId = params.dictId || 0;
        return await wasmDetectAruco(srcMat, dictId);
      }

      case 'qr_detector': {
        return await wasmDetectQR(srcMat);
      }

      case 'contour_area': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmContourArea(srcMat, thresholdValue);
      }

      case 'arc_length': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmArcLength(srcMat, thresholdValue);
      }

      case 'approx_poly_dp': {
        const thresholdValue = params.threshold || 127.0;
        const epsilon = params.epsilon || 5.0;
        return await wasmApproxPolyDP(srcMat, thresholdValue, epsilon);
      }

      // ==================== Batch 4: Advanced Filters, Transforms & Feature Detection ====================

      case 'anisotropic_diffusion': {
        const iterations = params.iterations || 10;
        const kappa = params.kappa || 20.0;
        const lambda = params.lambda || 0.25;
        return await wasmAnisotropicDiffusion(srcMat, iterations, kappa, lambda);
      }

      case 'morphology_tophat': {
        const ksize = params.ksize || 5;
        return await wasmMorphologyTophat(srcMat, ksize);
      }

      case 'morphology_blackhat': {
        const ksize = params.ksize || 5;
        return await wasmMorphologyBlackhat(srcMat, ksize);
      }

      case 'warp_perspective': {
        const angle = params.angle || 15.0;
        return await wasmWarpPerspective(srcMat, angle);
      }

      case 'get_rotation_matrix_2d': {
        const angle = params.angle || 45.0;
        const scale = params.scale || 1.0;
        return await wasmGetRotationMatrix2d(srcMat, angle, scale);
      }

      case 'normalize_histogram': {
        const alpha = params.alpha || 0.0;
        const beta = params.beta || 255.0;
        return await wasmNormalizeHistogram(srcMat, alpha, beta);
      }

      case 'compare_histograms': {
        return await wasmCompareHistograms(srcMat);
      }

      case 'back_projection': {
        return await wasmBackProjection(srcMat);
      }

      case 'moments': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmMoments(srcMat, thresholdValue);
      }

      case 'watershed': {
        return await wasmWatershed(srcMat);
      }

      case 'sift': {
        const nFeatures = params.n_features || 100;
        return await wasmSift(srcMat, nFeatures);
      }

      case 'orb': {
        const nFeatures = params.n_features || 100;
        return await wasmOrb(srcMat, nFeatures);
      }

      case 'brisk': {
        const threshold = params.threshold || 30;
        return await wasmBrisk(srcMat, threshold);
      }

      case 'akaze': {
        return await wasmAkaze(srcMat);
      }

      case 'kaze': {
        return await wasmKaze(srcMat);
      }

      // ==================== Batch 5: Advanced Features & Operations ====================

      case 'log_filter': {
        const ksize = params.ksize || 5;
        const sigma = params.sigma || 1.5;
        return await wasmLogFilter(srcMat, ksize, sigma);
      }

      case 'cvt_color_lab': {
        return await wasmCvtColorLab(srcMat);
      }

      case 'cvt_color_ycrcb': {
        return await wasmCvtColorYCrCb(srcMat);
      }

      case 'draw_ellipse': {
        const cx = params.cx || srcMat.width / 2;
        const cy = params.cy || srcMat.height / 2;
        const width = params.width || 100;
        const height = params.height || 60;
        const angle = params.angle || 0;
        const thickness = params.thickness || 2;
        return await wasmDrawEllipse(srcMat, cx, cy, width, height, angle, thickness);
      }

      case 'draw_polylines': {
        return await wasmDrawPolylines(srcMat);
      }

      case 'put_text': {
        const text = params.text || "OpenCV Rust";
        const x = params.x || 50;
        const y = params.y || 100;
        const font_scale = params.font_scale || 1.0;
        return await wasmPutText(srcMat, text, x, y, font_scale);
      }

      case 'min_enclosing_circle': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmMinEnclosingCircle(srcMat, thresholdValue);
      }

      case 'convex_hull': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmConvexHull(srcMat, thresholdValue);
      }

      case 'hu_moments': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmHuMoments(srcMat, thresholdValue);
      }

      case 'match_shapes': {
        const thresholdValue = params.threshold || 127.0;
        return await wasmMatchShapes(srcMat, thresholdValue);
      }

      case 'inpaint': {
        const radius = params.radius || 3;
        return await wasmInpaint(srcMat, radius);
      }

      case 'kmeans': {
        const k = params.k || 4;
        return await wasmKmeans(srcMat, k);
      }

      case 'find_homography': {
        const nFeatures = params.n_features || 100;
        return await wasmFindHomography(srcMat, nFeatures);
      }

      case 'brute_force_matcher': {
        const nFeatures = params.n_features || 100;
        return await wasmBruteForceMatcher(srcMat, nFeatures);
      }

      case 'hog_descriptor': {
        return await wasmHogDescriptor(srcMat);
      }

      case 'bg_subtractor_mog2': {
        const learningRate = params.learning_rate || 0.01;
        return await wasmBgSubtractorMog2(srcMat, learningRate);
      }

      case 'bg_subtractor_knn': {
        const learningRate = params.learning_rate || 0.01;
        return await wasmBgSubtractorKnn(srcMat, learningRate);
      }

      case 'farneback_optical_flow': {
        return await wasmFarnebackOpticalFlow(srcMat);
      }

      case 'tonemap_drago': {
        const bias = params.bias || 0.85;
        return await wasmTonemapDrago(srcMat, bias);
      }

      case 'tonemap_reinhard': {
        return await wasmTonemapReinhard(srcMat);
      }

      case 'meanshift_tracker': {
        return await wasmMeanshiftTracker(srcMat);
      }

      case 'camshift_tracker': {
        return await wasmCamshiftTracker(srcMat);
      }

      case 'mosse_tracker': {
        return await wasmMosseTracker(srcMat);
      }

      case 'csrt_tracker': {
        return await wasmCsrtTracker(srcMat);
      }

      case 'svm_classifier': {
        return await wasmSvmClassifier(srcMat);
      }

      case 'decision_tree': {
        return await wasmDecisionTree(srcMat);
      }

      case 'random_forest': {
        const nTrees = params.n_trees || 10;
        return await wasmRandomForest(srcMat, nTrees);
      }

      case 'knn': {
        const k = params.k || 5;
        return await wasmKnn(srcMat, k);
      }

      case 'neural_network': {
        return await wasmNeuralNetwork(srcMat);
      }

      case 'fast_nl_means': {
        const h = params.h || 10.0;
        const templateWindowSize = params.template_window_size || 7;
        const searchWindowSize = params.search_window_size || 21;
        return await wasmFastNlMeans(srcMat, h, templateWindowSize, searchWindowSize);
      }

      case 'super_resolution': {
        const scale = params.scale || 2.0;
        return await wasmSuperResolution(srcMat, scale);
      }

      case 'merge_debevec': {
        return await wasmMergeDebevec(srcMat);
      }

      case 'calibrate_camera': {
        return await wasmCalibrateCamera(srcMat);
      }

      case 'fisheye_calibration': {
        return await wasmFisheyeCalibration(srcMat);
      }

      case 'solve_pnp': {
        return await wasmSolvePnp(srcMat);
      }

      case 'stereo_calibration': {
        return await wasmStereoCalibration(srcMat);
      }

      case 'compute_disparity': {
        return await wasmComputeDisparity(srcMat);
      }

      case 'cascade_classifier': {
        return await wasmCascadeClassifier(srcMat);
      }

      case 'panorama_stitcher': {
        return await wasmPanoramaStitcher(srcMat);
      }

      case 'feather_blender': {
        const blendStrength = params.blend_strength || 0.5;
        return await wasmFeatherBlender(srcMat, blendStrength);
      }

      case 'stereo_rectification': {
        return await wasmStereoRectification(srcMat);
      }

      case 'multiband_blender': {
        const numBands = params.num_bands || 4;
        return await wasmMultibandBlender(srcMat, numBands);
      }

      case 'load_network': {
        return await wasmLoadNetwork(srcMat);
      }

      case 'blob_from_image': {
        return await wasmBlobFromImage(srcMat);
      }


      default:
        throw new Error(`Unknown demo: ${demoId}`);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>OpenCV-Rust Interactive Demos</h1>
          <div className="header-status">
            <span className={`status-badge ${wasmLoaded ? 'success' : 'warning'}`}>
              {wasmLoaded ? '✓ WASM Ready' : '⏳ Loading...'}
            </span>
            <span className={`status-badge ${gpuAvailable ? 'success' : 'warning'}`}>
              {gpuAvailable === null ? '⏳ GPU Init...' :
               gpuAvailable ? '✓ WebGPU' : '⚠ CPU Only'}
            </span>
          </div>
        </div>
      </header>

      <div className="app-content">
        <Sidebar />

        <main className="main-panel">
          <section className="controls-section">
            <DemoControls />
          </section>

          <section className="io-section">
            <InputOutput onProcess={processImage} />
          </section>

          <section className="metrics-section">
            <PerformanceMetrics />
          </section>

          <section className="history-section-container">
            <History />
          </section>
        </main>
      </div>

      <footer className="app-footer">
        <p>
          OpenCV-Rust v{wasmLoaded && getVersion()} • Pure Rust Image Processing •
          {gpuAvailable ? ' WebGPU Accelerated' : ' CPU Mode'}
        </p>
      </footer>
    </div>
  );
}

export default App;
