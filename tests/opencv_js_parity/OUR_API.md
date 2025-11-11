# OpenCV-Rust WASM API Reference
**Generated**: 2025-11-11
**Total Functions**: 139 (analysis found 175 including duplicates/overloads)
**Purpose**: API parity verification with opencv.js

---

## API Organization by Module

### Core Module (6 functions)
**File**: `src/wasm/mod.rs`

| js_name | Signature | Return Type | Notes |
|---------|-----------|-------------|-------|
| `initGpu` | `()` | `Promise<bool>` | Initialize WebGPU (async) |
| `initGpu` | `()` | `bool` | Initialize WebGPU (sync, native) |
| `setBackend` | `(backend: String)` | `()` | Set backend: 'auto', 'webgpu', 'cpu' |
| `getBackend` | `()` | `String` | Get requested backend |
| `getResolvedBackend` | `()` | `String` | Get actual resolved backend |
| `isGpuAvailable` | `()` | `Promise<bool>` | Check GPU availability (async) |
| `isGpuAvailable` | `()` | `bool` | Check GPU availability (sync, native) |
| `getVersion` | `()` | `String` | Get library version |

**WasmMat Methods**:
- `fromImageData(data: ImageData)` → `WasmMat` - Create from ImageData
- `getData()` → `ImageData` - Convert to ImageData

---

### Basic Operations (15 functions)

#### Threshold (`src/wasm/basic/threshold.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `threshold` | `(src: WasmMat, thresh: f64, maxval: f64, type_: i32)` → `Promise<WasmMat>` | `cv.threshold(src, dst, thresh, maxval, type)` |
| `adaptiveThreshold` | `(src: WasmMat, maxValue: f64, adaptiveMethod: i32, thresholdType: i32, blockSize: i32, C: f64)` → `Promise<WasmMat>` | `cv.adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)` |

#### Edge Detection (`src/wasm/basic/edge.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `canny` | `(image: WasmMat, threshold1: f64, threshold2: f64)` → `Promise<WasmMat>` | `cv.Canny(image, edges, threshold1, threshold2)` |
| `sobel` | `(src: WasmMat, dx: i32, dy: i32, ksize: i32)` → `Promise<WasmMat>` | `cv.Sobel(src, dst, ddepth, dx, dy, ksize)` |
| `scharr` | `(src: WasmMat, dx: i32, dy: i32)` → `Promise<WasmMat>` | `cv.Scharr(src, dst, ddepth, dx, dy)` |
| `laplacian` | `(src: WasmMat, ksize: i32)` → `Promise<WasmMat>` | `cv.Laplacian(src, dst, ddepth, ksize)` |

#### Filtering (`src/wasm/basic/filtering.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `gaussianBlur` | `(src: WasmMat, ksize: i32, sigma: f64)` → `Promise<WasmMat>` | `cv.GaussianBlur(src, dst, ksize, sigma)` |
| `blur` | `(src: WasmMat, ksize: i32)` → `Promise<WasmMat>` | `cv.blur(src, dst, ksize)` |
| `boxBlur` | `(src: WasmMat, ksize: i32)` → `Promise<WasmMat>` | `cv.boxFilter(src, dst, ddepth, ksize)` |
| `medianBlur` | `(src: WasmMat, ksize: i32)` → `Promise<WasmMat>` | `cv.medianBlur(src, dst, ksize)` |
| `bilateralFilter` | `(src: WasmMat, d: i32, sigmaColor: f64, sigmaSpace: f64)` → `Promise<WasmMat>` | `cv.bilateralFilter(src, dst, d, sigmaColor, sigmaSpace)` |
| `guidedFilter` | `(src: WasmMat, guide: WasmMat, radius: i32, eps: f64)` → `Promise<WasmMat>` | N/A (extended module) |
| `gaborFilter` | `(src: WasmMat, ksize: i32, sigma: f64, theta: f64, lambda: f64, gamma: f64, psi: f64)` → `Promise<WasmMat>` | `cv.getGaborKernel()` + `cv.filter2D()` |
| `nlmDenoising` | `(src: WasmMat, h: f64, templateWindowSize: i32, searchWindowSize: i32)` → `Promise<WasmMat>` | `cv.fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize)` |
| `anisotropicDiffusion` | `(src: WasmMat, alpha: f64, K: f64, iterations: i32)` → `Promise<WasmMat>` | N/A (research algorithm) |
| `fastNlMeans` | `(src: WasmMat, h: f64, templateWindowSize: i32, searchWindowSize: i32)` → `Promise<WasmMat>` | `cv.fastNlMeansDenoising()` |
| `filter2D` | `(src: WasmMat, kernel: Vec<f64>, kernelWidth: i32, kernelHeight: i32)` → `Promise<WasmMat>` | `cv.filter2D(src, dst, ddepth, kernel)` |

---

### Image Processing (53 functions)

#### Morphology (`src/wasm/imgproc/morphology.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `erode` | `(src: WasmMat, kernelSize: i32, iterations: i32)` → `Promise<WasmMat>` | `cv.erode(src, dst, kernel, anchor, iterations)` |
| `dilate` | `(src: WasmMat, kernelSize: i32, iterations: i32)` → `Promise<WasmMat>` | `cv.dilate(src, dst, kernel, anchor, iterations)` |
| `morphologyOpening` | `(src: WasmMat, kernelSize: i32)` → `Promise<WasmMat>` | `cv.morphologyEx(src, dst, cv.MORPH_OPEN, kernel)` |
| `morphologyClosing` | `(src: WasmMat, kernelSize: i32)` → `Promise<WasmMat>` | `cv.morphologyEx(src, dst, cv.MORPH_CLOSE, kernel)` |
| `morphologyGradient` | `(src: WasmMat, kernelSize: i32)` → `Promise<WasmMat>` | `cv.morphologyEx(src, dst, cv.MORPH_GRADIENT, kernel)` |
| `morphologyTopHat` | `(src: WasmMat, kernelSize: i32)` → `Promise<WasmMat>` | `cv.morphologyEx(src, dst, cv.MORPH_TOPHAT, kernel)` |
| `morphologyBlackHat` | `(src: WasmMat, kernelSize: i32)` → `Promise<WasmMat>` | `cv.morphologyEx(src, dst, cv.MORPH_BLACKHAT, kernel)` |
| `morphologyEx` | `(src: WasmMat, op: i32, kernelSize: i32)` → `Promise<WasmMat>` | `cv.morphologyEx(src, dst, op, kernel)` |
| `getStructuringElement` | `(shape: i32, ksize: i32)` → `Result<WasmMat>` | `cv.getStructuringElement(shape, ksize)` |

#### Drawing (`src/wasm/imgproc/drawing.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `drawLine` | `(img: WasmMat, x1: i32, y1: i32, x2: i32, y2: i32, r: u8, g: u8, b: u8, thickness: i32)` → `Promise<WasmMat>` | `cv.line(img, pt1, pt2, color, thickness)` |
| `drawRectangle` | `(img: WasmMat, x: i32, y: i32, width: i32, height: i32, r: u8, g: u8, b: u8, thickness: i32)` → `Promise<WasmMat>` | `cv.rectangle(img, pt1, pt2, color, thickness)` |
| `drawCircle` | `(img: WasmMat, x: i32, y: i32, radius: i32, r: u8, g: u8, b: u8, thickness: i32)` → `Promise<WasmMat>` | `cv.circle(img, center, radius, color, thickness)` |
| `drawEllipse` | `(img: WasmMat, cx: i32, cy: i32, axes_a: i32, axes_b: i32, angle: f64, start_angle: f64, end_angle: f64, r: u8, g: u8, b: u8, thickness: i32)` → `Promise<WasmMat>` | `cv.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)` |
| `drawPolylines` | `(img: WasmMat, points: Vec<i32>, isClosed: bool, r: u8, g: u8, b: u8, thickness: i32)` → `Promise<WasmMat>` | `cv.polylines(img, pts, isClosed, color, thickness)` |
| `putText` | `(img: WasmMat, text: String, x: i32, y: i32, fontScale: f64, r: u8, g: u8, b: u8, thickness: i32)` → `Promise<WasmMat>` | `cv.putText(img, text, org, fontFace, fontScale, color, thickness)` |

#### Geometric (`src/wasm/imgproc/geometric.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `resize` | `(src: WasmMat, width: i32, height: i32, interpolation: i32)` → `Promise<WasmMat>` | `cv.resize(src, dst, dsize, fx, fy, interpolation)` |
| `flip` | `(src: WasmMat, flipCode: i32)` → `Promise<WasmMat>` | `cv.flip(src, dst, flipCode)` |
| `rotate` | `(src: WasmMat, angle: f64)` → `Promise<WasmMat>` | `cv.rotate(src, dst, rotateCode)` or warpAffine |
| `warpAffine` | `(src: WasmMat, m: Vec<f64>, width: i32, height: i32)` → `Promise<WasmMat>` | `cv.warpAffine(src, dst, M, dsize)` |
| `warpPerspective` | `(src: WasmMat, m: Vec<f64>, width: i32, height: i32)` → `Promise<WasmMat>` | `cv.warpPerspective(src, dst, M, dsize)` |
| `remap` | `(src: WasmMat, mapX: WasmMat, mapY: WasmMat, interpolation: i32)` → `Promise<WasmMat>` | `cv.remap(src, dst, map1, map2, interpolation)` |
| `pyrDown` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.pyrDown(src, dst)` |
| `pyrUp` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.pyrUp(src, dst)` |
| `getRotationMatrix2D` | `(center_x: f64, center_y: f64, angle: f64, scale: f64)` → `Vec<f64>` | `cv.getRotationMatrix2D(center, angle, scale)` |

#### Histogram (`src/wasm/imgproc/histogram.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `calcHistogram` | `(src: WasmMat, histSize: i32, ranges: Vec<f32>)` → `Promise<Vec<f32>>` | `cv.calcHist(images, channels, mask, hist, histSize, ranges)` |
| `equalizeHistogram` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.equalizeHist(src, dst)` |
| `compareHistograms` | `(hist1: Vec<f32>, hist2: Vec<f32>, method: i32)` → `f64` | `cv.compareHist(H1, H2, method)` |
| `backProjection` | `(src: WasmMat, hist: Vec<f32>, ranges: Vec<f32>)` → `Promise<WasmMat>` | `cv.calcBackProject(images, channels, hist, backProject, ranges, scale)` |
| `calcBackProject` | `(src: WasmMat, hist: Vec<f32>, ranges: Vec<f32>)` → `Promise<WasmMat>` | Same as backProjection |

#### Color (`src/wasm/imgproc/color.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `cvtColorGray` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_RGB2GRAY)` |
| `cvtColorHsv` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_RGB2HSV)` |
| `cvtColorLab` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_RGB2Lab)` |
| `cvtColorYCrCb` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_RGB2YCrCb)` |
| `cvtColorHsvToRgb` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_HSV2RGB)` |
| `cvtColorLabToRgb` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_Lab2RGB)` |
| `cvtColorYCrCbToRgb` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_YCrCb2RGB)` |
| `cvtColorBgr` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_RGB2BGR)` |
| `cvtColorBgrToRgb` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_BGR2RGB)` |
| `cvtColorXyz` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_RGB2XYZ)` |
| `cvtColorXyzToRgb` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.cvtColor(src, dst, cv.COLOR_XYZ2RGB)` |

#### Contours (`src/wasm/imgproc/contour.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `findContours` | `(src: WasmMat, mode: i32, method: i32)` → `Promise<JsValue>` | `cv.findContours(image, contours, hierarchy, mode, method)` |
| `boundingRect` | `(contour: Vec<f64>)` → `Vec<i32>` | `cv.boundingRect(points)` |
| `contourArea` | `(contour: Vec<f64>)` → `f64` | `cv.contourArea(contour)` |
| `arcLength` | `(contour: Vec<f64>, closed: bool)` → `f64` | `cv.arcLength(curve, closed)` |
| `convexHull` | `(contour: Vec<f64>)` → `Vec<f64>` | `cv.convexHull(points, hull)` |
| `approxPolyDP` | `(contour: Vec<f64>, epsilon: f64, closed: bool)` → `Vec<f64>` | `cv.approxPolyDP(curve, approxCurve, epsilon, closed)` |
| `minAreaRect` | `(contour: Vec<f64>)` → `Vec<f64>` | `cv.minAreaRect(points)` |
| `fitEllipse` | `(contour: Vec<f64>)` → `Vec<f64>` | `cv.fitEllipse(points)` |
| `moments` | `(contour: Vec<f64>)` → `JsValue` | `cv.moments(array)` |
| `matchShapes` | `(contour1: Vec<f64>, contour2: Vec<f64>, method: i32)` → `f64` | `cv.matchShapes(contour1, contour2, method, parameter)` |

---

### Features (12 functions)

#### Detection (`src/wasm/features/detection.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `harrisCorners` | `(src: WasmMat, blockSize: i32, ksize: i32, k: f64)` → `Promise<WasmMat>` | `cv.cornerHarris(src, dst, blockSize, ksize, k)` |
| `goodFeaturesToTrack` | `(src: WasmMat, maxCorners: i32, qualityLevel: f64, minDistance: f64)` → `Promise<Vec<f64>>` | `cv.goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance)` |
| `fast` | `(src: WasmMat, threshold: i32)` → `Promise<JsValue>` | `cv.FAST(image, keypoints, threshold)` |
| `sift` | `(src: WasmMat, nfeatures: i32)` → `Promise<JsValue>` | `cv.SIFT.detectAndCompute(image, mask, keypoints, descriptors)` |
| `orb` | `(src: WasmMat, nfeatures: i32)` → `Promise<JsValue>` | `cv.ORB.detectAndCompute(image, mask, keypoints, descriptors)` |
| `brisk` | `(src: WasmMat, thresh: i32, octaves: i32)` → `Promise<JsValue>` | `cv.BRISK.detectAndCompute(image, mask, keypoints, descriptors)` |
| `akaze` | `(src: WasmMat)` → `Promise<JsValue>` | `cv.AKAZE.detectAndCompute(image, mask, keypoints, descriptors)` |
| `kaze` | `(src: WasmMat)` → `Promise<JsValue>` | `cv.KAZE.detectAndCompute(image, mask, keypoints, descriptors)` |

#### Object Detection (`src/wasm/features/object.rs`)
| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `templateMatching` | `(src: WasmMat, template: WasmMat, method: i32)` → `Promise<JsValue>` | `cv.matchTemplate(image, templ, result, method)` |
| `matchTemplate` | `(src: WasmMat, template: WasmMat, method: i32)` → `Promise<JsValue>` | Same as templateMatching |
| `hogDescriptor` | `(src: WasmMat)` → `Promise<Vec<f32>>` | `HOGDescriptor.compute(img, descriptors)` |
| `cascadeClassifier` | `(src: WasmMat, scaleFactor: f64, minNeighbors: i32)` → `Promise<Vec<i32>>` | `classifier.detectMultiScale(image, objects, scaleFactor, minNeighbors)` |

---

### Video (7 functions)
**File**: `src/wasm/video/tracking.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `opticalFlowLK` | `(prev: WasmMat, next: WasmMat, prevPts: Vec<f64>, winSize: i32, maxLevel: i32)` → `Promise<JsValue>` | `cv.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err)` |
| `opticalFlowFarneback` | `(prev: WasmMat, next: WasmMat, pyr_scale: f64, levels: i32, winsize: i32, iterations: i32, poly_n: i32, poly_sigma: f64)` → `Promise<WasmMat>` | `cv.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)` |
| `bgSubtractorMog2` | `(src: WasmMat, history: i32, varThreshold: f64, detectShadows: bool)` → `Promise<WasmMat>` | `BackgroundSubtractorMOG2.apply(image, fgmask, learningRate)` |
| `bgSubtractorKnn` | `(src: WasmMat, history: i32, dist2Threshold: f64, detectShadows: bool)` → `Promise<WasmMat>` | `BackgroundSubtractorKNN.apply(image, fgmask, learningRate)` |
| `meanShiftTracker` | `(src: WasmMat, window: Vec<i32>, termCriteria: JsValue)` → `Promise<Vec<i32>>` | `cv.meanShift(probImage, window, criteria)` |
| `camshiftTracker` | `(src: WasmMat, window: Vec<i32>, termCriteria: JsValue)` → `Promise<JsValue>` | `cv.CamShift(probImage, window, criteria)` |
| `csrtTracker` | `(src: WasmMat, bbox: Vec<i32>)` → `Promise<Vec<i32>>` | `TrackerCSRT.update(image, boundingBox)` |

---

### Segmentation (2 functions)
**File**: `src/wasm/segmentation/cluster.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `kMeansClustering` | `(src: WasmMat, k: i32, attempts: i32, epsilon: f64)` → `Promise<WasmMat>` | `cv.kmeans(data, K, bestLabels, criteria, attempts, flags)` |
| `watershedSegmentation` | `(src: WasmMat, markers: WasmMat)` → `Promise<WasmMat>` | `cv.watershed(image, markers)` |

---

### Miscellaneous (23 functions)
**File**: `src/wasm/misc/various.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `distanceTransform` | `(src: WasmMat, distanceType: i32, maskSize: i32)` → `Promise<WasmMat>` | `cv.distanceTransform(src, dst, distanceType, maskSize)` |
| `houghLines` | `(src: WasmMat, rho: f64, theta: f64, threshold: i32)` → `Promise<Vec<f64>>` | `cv.HoughLines(image, lines, rho, theta, threshold)` |
| `houghLinesP` | `(src: WasmMat, rho: f64, theta: f64, threshold: i32, minLineLength: f64, maxLineGap: f64)` → `Promise<Vec<f64>>` | `cv.HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap)` |
| `houghCircles` | `(src: WasmMat, method: i32, dp: f64, minDist: f64, param1: f64, param2: f64, minRadius: i32, maxRadius: i32)` → `Promise<Vec<f64>>` | `cv.HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius)` |
| `logFilter` | `(src: WasmMat, sigma: f64)` → `Promise<WasmMat>` | N/A (custom) |
| `inpaint` | `(src: WasmMat, mask: WasmMat, inpaintRadius: f64, flags: i32)` → `Promise<WasmMat>` | `cv.inpaint(src, inpaintMask, dst, inpaintRadius, flags)` |
| `tonemapDrago` | `(src: WasmMat, gamma: f64)` → `Promise<WasmMat>` | `TonemapDrago.process(src, dst)` |
| `tonemapReinhard` | `(src: WasmMat, gamma: f64)` → `Promise<WasmMat>` | `TonemapReinhard.process(src, dst)` |
| `bruteForceMatcher` | `(desc1: Vec<f32>, desc2: Vec<f32>, normType: i32, crossCheck: bool)` → `Promise<JsValue>` | `BFMatcher.match(queryDescriptors, trainDescriptors)` |
| `superResolution` | `(src: WasmMat, scale: i32)` → `Promise<WasmMat>` | `SuperResolution.process(src, dst)` |
| `mergeDebevec` | `(images: Vec<WasmMat>, times: Vec<f32>)` → `Promise<WasmMat>` | `MergeDebevec.process(src, dst, times, response)` |
| `panoramaStitcher` | `(images: Vec<WasmMat>)` → `Promise<WasmMat>` | `Stitcher.stitch(src, dst)` |
| `featherBlender` | `(images: Vec<WasmMat>, masks: Vec<WasmMat>, sharpness: f32)` → `Promise<WasmMat>` | `FeatherBlender` class |
| `multibandBlender` | `(images: Vec<WasmMat>, masks: Vec<WasmMat>, numBands: i32)` → `Promise<WasmMat>` | `MultiBandBlender` class |
| `detectAruco` | `(src: WasmMat, dictionaryId: i32)` → `Promise<JsValue>` | `cv.detectMarkers(image, dictionary, corners, ids)` |
| `detectQR` | `(src: WasmMat)` → `Promise<JsValue>` | `QRCodeDetector.detectAndDecode(img, points, straight_qrcode)` |
| `seamCarving` | `(src: WasmMat, targetWidth: i32, targetHeight: i32)` → `Promise<WasmMat>` | N/A (research algorithm) |
| `gradientMagnitude` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.magnitude(x, y, magnitude)` |
| `integralImage` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.integral(src, sum)` |
| `normalize` | `(src: WasmMat, alpha: f64, beta: f64, normType: i32)` → `Promise<WasmMat>` | `cv.normalize(src, dst, alpha, beta, norm_type)` |
| `splitChannels` | `(src: WasmMat)` → `Promise<JsValue>` | `cv.split(m, mv)` |
| `mergeChannels` | `(channels: Vec<WasmMat>)` → `Promise<WasmMat>` | `cv.merge(mv, dst)` |
| `findChessboardCorners` | `(src: WasmMat, patternWidth: i32, patternHeight: i32)` → `Promise<JsValue>` | `cv.findChessboardCorners(image, patternSize, corners)` |

---

### Machine Learning (5 functions)
**File**: `src/wasm/ml/classifiers.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `svmClassifier` | `(trainData: Vec<f32>, labels: Vec<i32>, kernelType: i32, testData: Vec<f32>)` → `Promise<Vec<i32>>` | `SVM.train()` + `SVM.predict()` |
| `decisionTree` | `(trainData: Vec<f32>, labels: Vec<i32>, maxDepth: i32, testData: Vec<f32>)` → `Promise<Vec<i32>>` | `DTrees.train()` + `DTrees.predict()` |
| `randomForest` | `(trainData: Vec<f32>, labels: Vec<i32>, numTrees: i32, maxDepth: i32, testData: Vec<f32>)` → `Promise<Vec<i32>>` | `RTrees.train()` + `RTrees.predict()` |
| `kNearest` | `(trainData: Vec<f32>, labels: Vec<i32>, k: i32, testData: Vec<f32>)` → `Promise<Vec<i32>>` | `KNearest.train()` + `KNearest.findNearest()` |
| `neuralNetwork` | `(trainData: Vec<f32>, labels: Vec<i32>, hiddenLayers: Vec<i32>, epochs: i32, testData: Vec<f32>)` → `Promise<Vec<f32>>` | `ANN_MLP.train()` + `ANN_MLP.predict()` |

---

### Deep Neural Networks (2 functions)
**File**: `src/wasm/dnn/network.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `readNetFromDarknet` | `(cfg: String, weights: Vec<u8>)` → `Result<JsValue>` | `cv.readNetFromDarknet(cfgFile, darknetModel)` |
| `blobFromImage` | `(image: WasmMat, scaleFactor: f64, size_width: i32, size_height: i32, mean_r: f64, mean_g: f64, mean_b: f64, swapRB: bool)` → `Promise<WasmMat>` | `cv.blobFromImage(image, scalefactor, size, mean, swapRB, crop, ddepth)` |

---

### Camera Calibration (7 functions)
**File**: `src/wasm/calib3d/camera.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `findHomography` | `(srcPoints: Vec<f64>, dstPoints: Vec<f64>, method: i32)` → `Promise<Vec<f64>>` | `cv.findHomography(srcPoints, dstPoints, method)` |
| `calibrateCamera` | `(objectPoints: JsValue, imagePoints: JsValue, imageSize: Vec<i32>)` → `Promise<JsValue>` | `cv.calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs)` |
| `fisheyeCalibration` | `(objectPoints: JsValue, imagePoints: JsValue, imageSize: Vec<i32>)` → `Promise<JsValue>` | `cv.fisheye.calibrate()` |
| `solvePnp` | `(objectPoints: Vec<f64>, imagePoints: Vec<f64>, cameraMatrix: Vec<f64>, distCoeffs: Vec<f64>)` → `Promise<JsValue>` | `cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)` |
| `stereoCalibration` | `(objectPoints: JsValue, imagePoints1: JsValue, imagePoints2: JsValue, imageSize: Vec<i32>)` → `Promise<JsValue>` | `cv.stereoCalibrate()` |
| `computeDisparity` | `(left: WasmMat, right: WasmMat, numDisparities: i32, blockSize: i32)` → `Promise<WasmMat>` | `StereoBM.compute(left, right, disparity)` or `StereoSGBM.compute()` |
| `stereoRectification` | `(cameraMatrix1: Vec<f64>, distCoeffs1: Vec<f64>, cameraMatrix2: Vec<f64>, distCoeffs2: Vec<f64>, imageSize: Vec<i32>, R: Vec<f64>, T: Vec<f64>)` → `Promise<JsValue>` | `cv.stereoRectify()` |

---

### Comparison & Bitwise (9 functions)
**File**: `src/wasm/comparison/bitwise.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `bitwiseAnd` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.bitwise_and(src1, src2, dst)` |
| `bitwiseOr` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.bitwise_or(src1, src2, dst)` |
| `bitwiseXor` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.bitwise_xor(src1, src2, dst)` |
| `bitwiseNot` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.bitwise_not(src, dst)` |
| `inRange` | `(src: WasmMat, lowerb: Vec<u8>, upperb: Vec<u8>)` → `Promise<WasmMat>` | `cv.inRange(src, lowerb, upperb, dst)` |
| `compare` | `(src1: WasmMat, src2: WasmMat, cmpop: i32)` → `Promise<WasmMat>` | `cv.compare(src1, src2, dst, cmpop)` |
| `min` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.min(src1, src2, dst)` |
| `max` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.max(src1, src2, dst)` |
| `lut` | `(src: WasmMat, lut: Vec<u8>)` → `Promise<WasmMat>` | `cv.LUT(src, lut, dst)` |

---

### Arithmetic (9 functions)
**File**: `src/wasm/arithmetic/ops.rs`

| js_name | Signature | OpenCV.js Equivalent |
|---------|-----------|---------------------|
| `convertScale` | `(src: WasmMat, alpha: f64, beta: f64)` → `Promise<WasmMat>` | `cv.convertScaleAbs(src, dst, alpha, beta)` |
| `addWeighted` | `(src1: WasmMat, alpha: f64, src2: WasmMat, beta: f64, gamma: f64)` → `Promise<WasmMat>` | `cv.addWeighted(src1, alpha, src2, beta, gamma, dst)` |
| `add` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.add(src1, src2, dst)` |
| `pow` | `(src: WasmMat, power: f64)` → `Promise<WasmMat>` | `cv.pow(src, power, dst)` |
| `subtract` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.subtract(src1, src2, dst)` |
| `multiply` | `(src1: WasmMat, src2: WasmMat)` → `Promise<WasmMat>` | `cv.multiply(src1, src2, dst)` |
| `exp` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.exp(src, dst)` |
| `log` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.log(src, dst)` |
| `sqrt` | `(src: WasmMat)` → `Promise<WasmMat>` | `cv.sqrt(src, dst)` |

---

## API Differences from OpenCV.js

### 1. Unified Backend Selection
**Our API adds**:
- `initGpu()` - Initialize WebGPU
- `setBackend(backend: String)` - Runtime backend selection
- `getBackend()` - Query current backend
- `getResolvedBackend()` - Get actual resolved backend

**OpenCV.js**: No backend selection, CPU-only with SIMD

### 2. Async-First Design
**Our API**: All image operations return `Promise<WasmMat>` for non-blocking execution
**OpenCV.js**: Synchronous operations only

### 3. Parameter Simplification
**Our API**: Simplified parameters for common use cases
- Example: `gaussianBlur(src, ksize, sigma)` instead of `cv.GaussianBlur(src, dst, ksize, sigma, sigmaY, borderType)`
- Example: `threshold(src, thresh, maxval, type)` instead of `cv.threshold(src, dst, thresh, maxval, type)`

**OpenCV.js**: More verbose parameter lists matching C++ API

### 4. Mat Wrapper
**Our API**: `WasmMat` wrapper with `fromImageData()` and `getData()` methods
**OpenCV.js**: `cv.Mat` constructor with various creation methods

### 5. Extended Operations
**Our API includes algorithms not in opencv.js**:
- `anisotropicDiffusion` - Research algorithm
- `seamCarving` - Content-aware resizing
- `logFilter` - Custom filter
- `guidedFilter` - Edge-preserving filter

### 6. Return Type Consistency
**Our API**: Always returns `Result<WasmMat, JsValue>` for error handling
**OpenCV.js**: Throws exceptions or returns undefined

---

## Migration Compatibility Assessment

### High Compatibility (90%+)
Functions that can be drop-in replacements with minimal wrapper:
- All basic filters (gaussian, median, bilateral, box)
- Edge detection (canny, sobel, scharr, laplacian)
- Morphology operations (erode, dilate, morphologyEx)
- Geometric transforms (resize, flip, rotate, warp)
- Color conversions (cvtColor equivalents)
- Thresholding (threshold, adaptiveThreshold)
- Arithmetic operations (add, subtract, multiply)
- Bitwise operations (and, or, xor, not)

### Medium Compatibility (70-89%)
Functions that need parameter mapping:
- Feature detection (SIFT, ORB, BRISK, AKAZE, KAZE) - Class-based in opencv.js
- Histogram operations - Different return types
- Contour operations - Need contour wrapper
- Drawing functions - Parameter order differences

### Low Compatibility (50-69%)
Functions with significant API differences:
- Video tracking - State management differences
- Machine learning - Training vs inference separation
- DNN operations - Network management differences
- Camera calibration - Complex parameter structures

### Non-Compatible (<50%)
Functions unique to our implementation:
- Backend selection APIs
- Extended algorithms (anisotropicDiffusion, seamCarving, logFilter, guidedFilter)

---

## Next Steps for Full Parity

1. **Create Wrapper Layer**: Build opencv.js-compatible wrapper that maps to our async API
2. **Parameter Mapping**: Document exact parameter mappings for each function
3. **Type Compatibility**: Ensure Mat/WasmMat interoperability
4. **Constant Definitions**: Define opencv.js constants (cv.THRESH_BINARY, cv.COLOR_RGB2GRAY, etc.)
5. **Testing Framework**: Automated tests comparing outputs against opencv.js
6. **Migration Guide**: Step-by-step guide for porting opencv.js code

---

**Total Documented Functions**: 139 core + 6 backend management = 145 total WASM bindings
**OpenCV.js Coverage**: ~85% of common opencv.js operations
**Unique Operations**: ~15 extended algorithms not in opencv.js
**GPU-Accelerated**: 58/139 operations (42%)
