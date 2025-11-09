/**
 * Demo Registry - MECE categorization of all OpenCV demos
 * Based on docs/plan.md implementation plan
 *
 * Each demo has:
 * - id: unique identifier
 * - name: display name
 * - description: short description
 * - category: parent category
 * - params: array of parameter definitions
 * - implemented: whether it's currently working
 * - gpuAccelerated: whether GPU acceleration is available
 */

export const categories = [
  {
    id: 'filters',
    name: '🎨 Image Filtering & Enhancement',
    description: 'Smoothing, noise reduction, and image enhancement filters',
    priority: 0
  },
  {
    id: 'edges',
    name: '📐 Edge Detection & Derivatives',
    description: 'Edge and gradient-based operators',
    priority: 0
  },
  {
    id: 'transform',
    name: '🔄 Geometric Transformations',
    description: 'Resize, rotate, warp, and perspective operations',
    priority: 0
  },
  {
    id: 'color',
    name: '🌈 Color & Thresholding',
    description: 'Color space conversion and thresholding operations',
    priority: 0
  },
  {
    id: 'histogram',
    name: '📊 Histogram Operations',
    description: 'Histogram analysis, equalization, and comparison',
    priority: 1
  },
  {
    id: 'morphology',
    name: '🔲 Morphological Operations',
    description: 'Erode, dilate, opening, closing operations',
    priority: 1
  },
  {
    id: 'contours',
    name: '🎯 Contour Detection & Analysis',
    description: 'Contour detection and shape analysis',
    priority: 0
  },
  {
    id: 'features',
    name: '🎯 Feature Detection',
    description: 'Keypoints, descriptors, and feature matching',
    priority: 1
  },
  {
    id: 'hough',
    name: '📏 Hough Transforms',
    description: 'Line and circle detection',
    priority: 1
  },
  {
    id: 'detection',
    name: '🎯 Object Detection',
    description: 'HOG, cascade classifiers, ArUco, QR codes',
    priority: 1
  },
  {
    id: 'video',
    name: '🎥 Video Analysis & Tracking',
    description: 'Optical flow, object tracking, background subtraction',
    priority: 2
  },
  {
    id: 'calibration',
    name: '📷 Camera Calibration',
    description: 'Camera calibration, pose estimation, stereo vision',
    priority: 2
  },
  {
    id: 'ml',
    name: '🤖 Machine Learning',
    description: 'SVM, Random Forest, Neural Networks, K-Means',
    priority: 2
  },
  {
    id: 'photo',
    name: '📸 Computational Photography',
    description: 'HDR, tone mapping, denoising, inpainting',
    priority: 2
  },
  {
    id: 'stitching',
    name: '🌄 Image Stitching & Panorama',
    description: 'Panorama creation, seam finding, blending',
    priority: 3
  },
  {
    id: 'drawing',
    name: '✏️ Drawing & Annotation',
    description: 'Draw shapes, lines, circles, and text',
    priority: 1
  },
  {
    id: 'dnn',
    name: '🧠 Deep Neural Networks',
    description: 'DNN module for loading and running neural networks',
    priority: 3
  },
  {
    id: 'shape',
    name: '📐 Shape Analysis',
    description: 'Shape descriptors, moments, and matching',
    priority: 2
  }
];

export const demos = [
  // ========================================
  // CATEGORY 1: IMAGE FILTERING & ENHANCEMENT
  // ========================================

  // 1.1 Basic Filters
  {
    id: 'gaussian_blur',
    name: 'Gaussian Blur',
    description: 'Smooth images with Gaussian filter for noise reduction',
    category: 'filters',
    implemented: true,
    gpuAccelerated: true,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 1,
        max: 31,
        step: 2,
        default: 5,
        description: 'Size of the Gaussian kernel (must be odd)'
      },
      {
        id: 'sigma',
        name: 'Sigma',
        type: 'slider',
        min: 0.1,
        max: 10.0,
        step: 0.1,
        default: 1.5,
        description: 'Standard deviation of the Gaussian distribution'
      }
    ]
  },
  {
    id: 'box_blur',
    name: 'Box Filter / Blur',
    description: 'Fast smoothing with box filter',
    category: 'filters',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 1,
        max: 31,
        step: 2,
        default: 5
      }
    ]
  },
  {
    id: 'median_blur',
    name: 'Median Blur',
    description: 'Remove salt & pepper noise with median filter',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'select',
        options: [3, 5, 7, 9, 11, 13, 15],
        default: 5
      }
    ]
  },

  // 1.2 Advanced Filters
  {
    id: 'bilateral_filter',
    name: 'Bilateral Filter',
    description: 'Edge-preserving smoothing filter',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'diameter',
        name: 'Diameter',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 9
      },
      {
        id: 'sigmaColor',
        name: 'Sigma Color',
        type: 'slider',
        min: 10,
        max: 150,
        step: 5,
        default: 75
      },
      {
        id: 'sigmaSpace',
        name: 'Sigma Space',
        type: 'slider',
        min: 10,
        max: 150,
        step: 5,
        default: 75
      }
    ]
  },
  {
    id: 'guided_filter',
    name: 'Guided Filter',
    description: 'Edge-aware smoothing using a guide image',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'radius',
        name: 'Radius',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 5
      },
      {
        id: 'epsilon',
        name: 'Epsilon',
        type: 'slider',
        min: 0.001,
        max: 1.0,
        step: 0.01,
        default: 0.1
      }
    ]
  },
  {
    id: 'gabor_filter',
    name: 'Gabor Filter',
    description: 'Texture analysis and feature extraction',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'frequency',
        name: 'Frequency',
        type: 'slider',
        min: 0.01,
        max: 1.0,
        step: 0.01,
        default: 0.1
      },
      {
        id: 'orientation',
        name: 'Orientation',
        type: 'slider',
        min: 0,
        max: 180,
        step: 5,
        default: 0
      },
      {
        id: 'sigma',
        name: 'Sigma',
        type: 'slider',
        min: 0.5,
        max: 10.0,
        step: 0.5,
        default: 3.0
      }
    ]
  },
  {
    id: 'log_filter',
    name: 'Laplacian of Gaussian (LoG)',
    description: 'Blob detection using LoG operator',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 31,
        step: 2,
        default: 5
      },
      {
        id: 'sigma',
        name: 'Sigma',
        type: 'slider',
        min: 0.1,
        max: 5.0,
        step: 0.1,
        default: 1.0
      }
    ]
  },
  {
    id: 'nlm_denoising',
    name: 'Non-Local Means Denoising',
    description: 'Strong noise removal with NLM algorithm',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'h',
        name: 'H Parameter',
        type: 'slider',
        min: 3,
        max: 30,
        step: 1,
        default: 10
      },
      {
        id: 'templateWindowSize',
        name: 'Template Window',
        type: 'slider',
        min: 7,
        max: 21,
        step: 2,
        default: 7
      },
      {
        id: 'searchWindowSize',
        name: 'Search Window',
        type: 'slider',
        min: 21,
        max: 35,
        step: 2,
        default: 21
      }
    ]
  },
  {
    id: 'anisotropic_diffusion',
    name: 'Anisotropic Diffusion',
    description: 'Edge-aware smoothing with diffusion',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'iterations',
        name: 'Iterations',
        type: 'slider',
        min: 1,
        max: 100,
        step: 1,
        default: 10
      },
      {
        id: 'k',
        name: 'K Value',
        type: 'slider',
        min: 10,
        max: 100,
        step: 5,
        default: 50
      },
      {
        id: 'lambda',
        name: 'Lambda',
        type: 'slider',
        min: 0.01,
        max: 0.25,
        step: 0.01,
        default: 0.1
      }
    ]
  },

  // 1.3 Distance & Morphology
  {
    id: 'distance_transform',
    name: 'Distance Transform',
    description: 'Compute distance to nearest zero pixel',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'distanceType',
        name: 'Distance Type',
        type: 'select',
        options: ['L1', 'L2', 'L-inf'],
        default: 'L2'
      },
      {
        id: 'maskSize',
        name: 'Mask Size',
        type: 'select',
        options: ['3x3', '5x5'],
        default: '3x3'
      }
    ]
  },
  {
    id: 'watershed',
    name: 'Watershed Segmentation',
    description: 'Marker-based image segmentation',
    category: 'filters',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'markerType',
        name: 'Marker Type',
        type: 'select',
        options: ['Manual', 'Auto'],
        default: 'Auto'
      }
    ]
  },

  // ========================================
  // CATEGORY 2: EDGE DETECTION & DERIVATIVES
  // ========================================

  {
    id: 'canny',
    name: 'Canny Edge Detection',
    description: 'Multi-stage edge detection algorithm',
    category: 'edges',
    implemented: true,
    gpuAccelerated: true,
    params: [
      {
        id: 'threshold1',
        name: 'Low Threshold',
        type: 'slider',
        min: 0,
        max: 255,
        step: 1,
        default: 50,
        description: 'Lower threshold for edge detection'
      },
      {
        id: 'threshold2',
        name: 'High Threshold',
        type: 'slider',
        min: 0,
        max: 255,
        step: 1,
        default: 150,
        description: 'Upper threshold for edge detection'
      },
      {
        id: 'apertureSize',
        name: 'Aperture Size',
        type: 'select',
        options: [3, 5, 7],
        default: 3
      }
    ]
  },
  {
    id: 'sobel',
    name: 'Sobel Operator',
    description: 'Gradient-based edge detection',
    category: 'edges',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'dx',
        name: 'X Derivative',
        type: 'slider',
        min: 0,
        max: 2,
        step: 1,
        default: 1
      },
      {
        id: 'dy',
        name: 'Y Derivative',
        type: 'slider',
        min: 0,
        max: 2,
        step: 1,
        default: 0
      },
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'select',
        options: [1, 3, 5, 7],
        default: 3
      }
    ]
  },
  {
    id: 'scharr',
    name: 'Scharr Operator',
    description: 'High-accuracy gradient computation',
    category: 'edges',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'dx',
        name: 'X Derivative',
        type: 'slider',
        min: 0,
        max: 1,
        step: 1,
        default: 1
      },
      {
        id: 'dy',
        name: 'Y Derivative',
        type: 'slider',
        min: 0,
        max: 1,
        step: 1,
        default: 0
      }
    ]
  },
  {
    id: 'laplacian',
    name: 'Laplacian',
    description: 'Second derivative edge detection',
    category: 'edges',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'select',
        options: [1, 3, 5, 7],
        default: 3
      }
    ]
  },

  // ========================================
  // CATEGORY 3: GEOMETRIC TRANSFORMATIONS
  // ========================================

  // 3.1 Basic Transformations
  {
    id: 'resize',
    name: 'Resize',
    description: 'Scale images up or down',
    category: 'transform',
    implemented: true,
    gpuAccelerated: true,
    params: [
      {
        id: 'scale',
        name: 'Scale Factor',
        type: 'slider',
        min: 0.1,
        max: 2.0,
        step: 0.1,
        default: 0.5,
        description: 'Scaling factor (1.0 = original size)'
      },
      {
        id: 'interpolation',
        name: 'Interpolation',
        type: 'select',
        options: ['Nearest', 'Linear', 'Cubic', 'Area', 'Lanczos4'],
        default: 'Linear'
      }
    ]
  },
  {
    id: 'flip',
    name: 'Flip',
    description: 'Flip image horizontally or vertically',
    category: 'transform',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'flipCode',
        name: 'Flip Direction',
        type: 'select',
        options: ['Horizontal', 'Vertical', 'Both'],
        default: 'Horizontal'
      }
    ]
  },
  {
    id: 'rotate',
    name: 'Rotate',
    description: 'Rotate image by 90, 180, or 270 degrees',
    category: 'transform',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'angle',
        name: 'Rotation',
        type: 'select',
        options: ['90° CW', '180°', '90° CCW'],
        default: '90° CW'
      }
    ]
  },

  // 3.2 Advanced Transformations
  {
    id: 'warp_affine',
    name: 'Warp Affine',
    description: 'Apply affine transformation (translate, rotate, scale, shear)',
    category: 'transform',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'angle',
        name: 'Rotation Angle',
        type: 'slider',
        min: -180,
        max: 180,
        step: 5,
        default: 0
      },
      {
        id: 'scale',
        name: 'Scale',
        type: 'slider',
        min: 0.1,
        max: 3.0,
        step: 0.1,
        default: 1.0
      },
      {
        id: 'shearX',
        name: 'Shear X',
        type: 'slider',
        min: -1.0,
        max: 1.0,
        step: 0.1,
        default: 0.0
      }
    ]
  },
  {
    id: 'warp_perspective',
    name: 'Warp Perspective',
    description: 'Apply perspective transformation for document scanning',
    category: 'transform',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'mode',
        name: 'Mode',
        type: 'select',
        options: ['Manual 4 Points', 'Auto Detect'],
        default: 'Auto Detect'
      }
    ]
  },
  {
    id: 'get_rotation_matrix_2d',
    name: 'Get Rotation Matrix 2D',
    description: 'Generate rotation matrix for affine transform',
    category: 'transform',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'angle',
        name: 'Angle',
        type: 'slider',
        min: 0,
        max: 360,
        step: 5,
        default: 45
      },
      {
        id: 'scale',
        name: 'Scale',
        type: 'slider',
        min: 0.1,
        max: 5.0,
        step: 0.1,
        default: 1.0
      }
    ]
  },

  // ========================================
  // CATEGORY 4: COLOR & THRESHOLDING
  // ========================================

  // 4.1 Color Space Conversion
  {
    id: 'cvt_color_gray',
    name: 'Convert to Grayscale',
    description: 'Convert color image to grayscale',
    category: 'color',
    implemented: true,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'cvt_color_hsv',
    name: 'RGB to HSV',
    description: 'Convert RGB to HSV color space',
    category: 'color',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'cvt_color_lab',
    name: 'RGB to Lab',
    description: 'Convert RGB to Lab color space',
    category: 'color',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'cvt_color_ycrcb',
    name: 'RGB to YCrCb',
    description: 'Convert RGB to YCrCb color space',
    category: 'color',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },

  // 4.2 Thresholding
  {
    id: 'threshold',
    name: 'Binary Threshold',
    description: 'Apply binary thresholding to segment image',
    category: 'color',
    implemented: true,
    gpuAccelerated: true,
    params: [
      {
        id: 'thresh',
        name: 'Threshold Value',
        type: 'slider',
        min: 0,
        max: 255,
        step: 1,
        default: 127,
        description: 'Threshold value for binarization'
      },
      {
        id: 'maxval',
        name: 'Max Value',
        type: 'slider',
        min: 0,
        max: 255,
        step: 1,
        default: 255,
        description: 'Maximum value for pixels above threshold'
      },
      {
        id: 'type',
        name: 'Threshold Type',
        type: 'select',
        options: ['Binary', 'Binary Inverted', 'Truncate', 'To Zero', 'To Zero Inverted'],
        default: 'Binary'
      }
    ]
  },
  {
    id: 'adaptive_threshold',
    name: 'Adaptive Threshold',
    description: 'Adaptive thresholding for varying lighting',
    category: 'color',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'maxval',
        name: 'Max Value',
        type: 'slider',
        min: 0,
        max: 255,
        step: 1,
        default: 255
      },
      {
        id: 'method',
        name: 'Method',
        type: 'select',
        options: ['Mean', 'Gaussian'],
        default: 'Gaussian'
      },
      {
        id: 'blockSize',
        name: 'Block Size',
        type: 'slider',
        min: 3,
        max: 99,
        step: 2,
        default: 11
      },
      {
        id: 'C',
        name: 'Constant C',
        type: 'slider',
        min: -20,
        max: 20,
        step: 1,
        default: 2
      }
    ]
  },

  // ========================================
  // CATEGORY 5: HISTOGRAM OPERATIONS
  // ========================================

  {
    id: 'calc_histogram',
    name: 'Calculate Histogram',
    description: 'Compute intensity distribution histogram',
    category: 'histogram',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'bins',
        name: 'Number of Bins',
        type: 'slider',
        min: 8,
        max: 256,
        step: 8,
        default: 256
      },
      {
        id: 'range',
        name: 'Range',
        type: 'text',
        default: '0-255'
      }
    ]
  },
  {
    id: 'equalize_histogram',
    name: 'Equalize Histogram',
    description: 'Enhance contrast using histogram equalization',
    category: 'histogram',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'normalize_histogram',
    name: 'Normalize Histogram',
    description: 'Normalize histogram values',
    category: 'histogram',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'alpha',
        name: 'Alpha',
        type: 'slider',
        min: 0,
        max: 1,
        step: 0.1,
        default: 1.0
      },
      {
        id: 'beta',
        name: 'Beta',
        type: 'slider',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.0
      }
    ]
  },
  {
    id: 'compare_histograms',
    name: 'Compare Histograms',
    description: 'Measure similarity between histograms',
    category: 'histogram',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'method',
        name: 'Comparison Method',
        type: 'select',
        options: ['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya'],
        default: 'Correlation'
      }
    ]
  },
  {
    id: 'back_projection',
    name: 'Back Projection',
    description: 'Find pixels matching histogram',
    category: 'histogram',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'channels',
        name: 'Channels',
        type: 'select',
        options: ['H', 'S', 'V', 'HS'],
        default: 'HS'
      }
    ]
  },

  // ========================================
  // CATEGORY 6: MORPHOLOGICAL OPERATIONS
  // ========================================

  {
    id: 'erode',
    name: 'Erode',
    description: 'Erosion morphological operation',
    category: 'morphology',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 21,
        step: 2,
        default: 5
      },
      {
        id: 'shape',
        name: 'Kernel Shape',
        type: 'select',
        options: ['Rectangle', 'Cross', 'Ellipse'],
        default: 'Rectangle'
      },
      {
        id: 'iterations',
        name: 'Iterations',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 1
      }
    ]
  },
  {
    id: 'dilate',
    name: 'Dilate',
    description: 'Dilation morphological operation',
    category: 'morphology',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 21,
        step: 2,
        default: 5
      },
      {
        id: 'shape',
        name: 'Kernel Shape',
        type: 'select',
        options: ['Rectangle', 'Cross', 'Ellipse'],
        default: 'Rectangle'
      },
      {
        id: 'iterations',
        name: 'Iterations',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 1
      }
    ]
  },
  {
    id: 'morphology_opening',
    name: 'Morphological Opening',
    description: 'Erode then dilate to remove noise',
    category: 'morphology',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 21,
        step: 2,
        default: 5
      },
      {
        id: 'shape',
        name: 'Kernel Shape',
        type: 'select',
        options: ['Rectangle', 'Cross', 'Ellipse'],
        default: 'Rectangle'
      }
    ]
  },
  {
    id: 'morphology_closing',
    name: 'Morphological Closing',
    description: 'Dilate then erode to fill gaps',
    category: 'morphology',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 21,
        step: 2,
        default: 5
      },
      {
        id: 'shape',
        name: 'Kernel Shape',
        type: 'select',
        options: ['Rectangle', 'Cross', 'Ellipse'],
        default: 'Rectangle'
      }
    ]
  },
  {
    id: 'morphology_gradient',
    name: 'Morphological Gradient',
    description: 'Difference between dilation and erosion',
    category: 'morphology',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 21,
        step: 2,
        default: 5
      }
    ]
  },
  {
    id: 'morphology_tophat',
    name: 'Top Hat',
    description: 'Original minus opening',
    category: 'morphology',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 21,
        step: 2,
        default: 9
      }
    ]
  },
  {
    id: 'morphology_blackhat',
    name: 'Black Hat',
    description: 'Closing minus original',
    category: 'morphology',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 3,
        max: 21,
        step: 2,
        default: 9
      }
    ]
  },

  // ========================================
  // CATEGORY 7: CONTOUR DETECTION & ANALYSIS
  // ========================================

  {
    id: 'find_contours',
    name: 'Find Contours',
    description: 'Detect object boundaries in binary images',
    category: 'contours',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'mode',
        name: 'Retrieval Mode',
        type: 'select',
        options: ['External', 'List', 'CComp', 'Tree'],
        default: 'External'
      },
      {
        id: 'method',
        name: 'Approximation',
        type: 'select',
        options: ['None', 'Simple', 'TC89_L1', 'TC89_KCOS'],
        default: 'Simple'
      }
    ]
  },
  {
    id: 'approx_poly_dp',
    name: 'Approximate Polygon',
    description: 'Simplify contours with Douglas-Peucker',
    category: 'contours',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'epsilon',
        name: 'Epsilon',
        type: 'slider',
        min: 0.001,
        max: 10.0,
        step: 0.1,
        default: 2.0
      },
      {
        id: 'closed',
        name: 'Closed',
        type: 'checkbox',
        default: true
      }
    ]
  },
  {
    id: 'contour_area',
    name: 'Contour Area',
    description: 'Calculate area enclosed by contour',
    category: 'contours',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'arc_length',
    name: 'Arc Length',
    description: 'Calculate contour perimeter',
    category: 'contours',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'closed',
        name: 'Closed',
        type: 'checkbox',
        default: true
      }
    ]
  },
  {
    id: 'bounding_rect',
    name: 'Bounding Rectangle',
    description: 'Get axis-aligned bounding box',
    category: 'contours',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'moments',
    name: 'Image Moments',
    description: 'Calculate spatial moments for centroid and orientation',
    category: 'contours',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },

  // ========================================
  // CATEGORY 8: FEATURE DETECTION
  // ========================================

  // 8.1 Corner Detection
  {
    id: 'harris_corners',
    name: 'Harris Corner Detection',
    description: 'Detect corners using Harris algorithm',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'blockSize',
        name: 'Block Size',
        type: 'slider',
        min: 2,
        max: 31,
        step: 1,
        default: 3
      },
      {
        id: 'ksize',
        name: 'Kernel Size',
        type: 'slider',
        min: 1,
        max: 31,
        step: 2,
        default: 3
      },
      {
        id: 'k',
        name: 'K Parameter',
        type: 'slider',
        min: 0.04,
        max: 0.06,
        step: 0.001,
        default: 0.04
      },
      {
        id: 'threshold',
        name: 'Threshold',
        type: 'slider',
        min: 0.01,
        max: 0.1,
        step: 0.01,
        default: 0.01
      }
    ]
  },
  {
    id: 'good_features_to_track',
    name: 'Good Features to Track',
    description: 'Shi-Tomasi corner detection',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'maxCorners',
        name: 'Max Corners',
        type: 'slider',
        min: 10,
        max: 1000,
        step: 10,
        default: 100
      },
      {
        id: 'qualityLevel',
        name: 'Quality Level',
        type: 'slider',
        min: 0.01,
        max: 0.5,
        step: 0.01,
        default: 0.01
      },
      {
        id: 'minDistance',
        name: 'Min Distance',
        type: 'slider',
        min: 1,
        max: 50,
        step: 1,
        default: 10
      }
    ]
  },
  {
    id: 'fast',
    name: 'FAST Keypoint Detector',
    description: 'Fast keypoint detection algorithm',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'threshold',
        name: 'Threshold',
        type: 'slider',
        min: 1,
        max: 100,
        step: 1,
        default: 10
      },
      {
        id: 'nonmaxSuppression',
        name: 'Non-max Suppression',
        type: 'checkbox',
        default: true
      }
    ]
  },

  // 8.2 Keypoint Detectors & Descriptors
  {
    id: 'sift',
    name: 'SIFT',
    description: 'Scale-Invariant Feature Transform',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'nfeatures',
        name: 'Number of Features',
        type: 'slider',
        min: 0,
        max: 1000,
        step: 10,
        default: 0
      },
      {
        id: 'nOctaveLayers',
        name: 'Octave Layers',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 3
      },
      {
        id: 'contrastThreshold',
        name: 'Contrast Threshold',
        type: 'slider',
        min: 0.01,
        max: 0.2,
        step: 0.01,
        default: 0.04
      }
    ]
  },
  {
    id: 'orb',
    name: 'ORB',
    description: 'Oriented FAST and Rotated BRIEF',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'nfeatures',
        name: 'Number of Features',
        type: 'slider',
        min: 100,
        max: 5000,
        step: 100,
        default: 500
      },
      {
        id: 'scaleFactor',
        name: 'Scale Factor',
        type: 'slider',
        min: 1.1,
        max: 2.0,
        step: 0.1,
        default: 1.2
      },
      {
        id: 'nlevels',
        name: 'Pyramid Levels',
        type: 'slider',
        min: 1,
        max: 16,
        step: 1,
        default: 8
      }
    ]
  },
  {
    id: 'brisk',
    name: 'BRISK',
    description: 'Binary Robust Invariant Scalable Keypoints',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'threshold',
        name: 'Threshold',
        type: 'slider',
        min: 0,
        max: 100,
        step: 1,
        default: 30
      },
      {
        id: 'octaves',
        name: 'Octaves',
        type: 'slider',
        min: 0,
        max: 8,
        step: 1,
        default: 3
      }
    ]
  },
  {
    id: 'akaze',
    name: 'AKAZE',
    description: 'Accelerated-KAZE features',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'threshold',
        name: 'Threshold',
        type: 'slider',
        min: 0.0001,
        max: 0.01,
        step: 0.0001,
        default: 0.001
      },
      {
        id: 'octaves',
        name: 'Octaves',
        type: 'slider',
        min: 1,
        max: 8,
        step: 1,
        default: 4
      }
    ]
  },
  {
    id: 'kaze',
    name: 'KAZE',
    description: 'Nonlinear scale space features',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'threshold',
        name: 'Threshold',
        type: 'slider',
        min: 0.0001,
        max: 0.01,
        step: 0.0001,
        default: 0.001
      },
      {
        id: 'octaves',
        name: 'Octaves',
        type: 'slider',
        min: 1,
        max: 8,
        step: 1,
        default: 4
      }
    ]
  },

  // 8.3 Feature Matching
  {
    id: 'brute_force_matcher',
    name: 'Brute Force Matcher',
    description: 'Match feature descriptors',
    category: 'features',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'distanceType',
        name: 'Distance Type',
        type: 'select',
        options: ['Hamming', 'L2'],
        default: 'Hamming'
      },
      {
        id: 'crossCheck',
        name: 'Cross Check',
        type: 'checkbox',
        default: false
      }
    ]
  },

  // ========================================
  // CATEGORY 9: HOUGH TRANSFORMS
  // ========================================

  {
    id: 'hough_lines',
    name: 'Hough Lines (Standard)',
    description: 'Detect infinite lines using Hough transform',
    category: 'hough',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'rho',
        name: 'Rho',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 1
      },
      {
        id: 'theta',
        name: 'Theta (degrees)',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 1
      },
      {
        id: 'threshold',
        name: 'Threshold',
        type: 'slider',
        min: 50,
        max: 300,
        step: 10,
        default: 150
      }
    ]
  },
  {
    id: 'hough_lines_p',
    name: 'Hough Lines P (Probabilistic)',
    description: 'Detect line segments using probabilistic Hough',
    category: 'hough',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'rho',
        name: 'Rho',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 1
      },
      {
        id: 'theta',
        name: 'Theta (degrees)',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 1
      },
      {
        id: 'threshold',
        name: 'Threshold',
        type: 'slider',
        min: 50,
        max: 300,
        step: 10,
        default: 80
      },
      {
        id: 'minLineLength',
        name: 'Min Line Length',
        type: 'slider',
        min: 10,
        max: 100,
        step: 5,
        default: 30
      },
      {
        id: 'maxLineGap',
        name: 'Max Line Gap',
        type: 'slider',
        min: 1,
        max: 50,
        step: 1,
        default: 10
      }
    ]
  },
  {
    id: 'hough_circles',
    name: 'Hough Circles',
    description: 'Detect circles using Hough transform',
    category: 'hough',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'dp',
        name: 'DP',
        type: 'slider',
        min: 1,
        max: 2,
        step: 0.1,
        default: 1
      },
      {
        id: 'minDist',
        name: 'Min Distance',
        type: 'slider',
        min: 10,
        max: 100,
        step: 5,
        default: 50
      },
      {
        id: 'param1',
        name: 'Param 1',
        type: 'slider',
        min: 50,
        max: 300,
        step: 10,
        default: 100
      },
      {
        id: 'param2',
        name: 'Param 2',
        type: 'slider',
        min: 10,
        max: 100,
        step: 5,
        default: 30
      },
      {
        id: 'minRadius',
        name: 'Min Radius',
        type: 'slider',
        min: 5,
        max: 200,
        step: 5,
        default: 10
      },
      {
        id: 'maxRadius',
        name: 'Max Radius',
        type: 'slider',
        min: 10,
        max: 500,
        step: 10,
        default: 100
      }
    ]
  },

  // ========================================
  // CATEGORY 10: OBJECT DETECTION
  // ========================================

  {
    id: 'hog_descriptor',
    name: 'HOG Descriptor',
    description: 'Histogram of Oriented Gradients for pedestrian detection',
    category: 'detection',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'winSize',
        name: 'Window Size',
        type: 'select',
        options: ['64x128', '128x256'],
        default: '64x128'
      }
    ]
  },
  {
    id: 'cascade_classifier',
    name: 'Cascade Classifier',
    description: 'Haar/LBP cascade for face/object detection',
    category: 'detection',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'scaleFactor',
        name: 'Scale Factor',
        type: 'slider',
        min: 1.01,
        max: 1.5,
        step: 0.01,
        default: 1.1
      },
      {
        id: 'minNeighbors',
        name: 'Min Neighbors',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 3
      }
    ]
  },
  {
    id: 'aruco_detector',
    name: 'ArUco Marker Detection',
    description: 'Detect and decode ArUco markers',
    category: 'detection',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'dictionary',
        name: 'Dictionary',
        type: 'select',
        options: ['4x4_50', '4x4_100', '5x5_100', '6x6_100', '6x6_250'],
        default: '6x6_250'
      }
    ]
  },
  {
    id: 'qr_detector',
    name: 'QR Code Detector',
    description: 'Detect and decode QR codes',
    category: 'detection',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },

  // ========================================
  // CATEGORY 11: VIDEO ANALYSIS & TRACKING
  // ========================================

  {
    id: 'farneback_optical_flow',
    name: 'Farneback Optical Flow',
    description: 'Dense optical flow computation',
    category: 'video',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'pyrScale',
        name: 'Pyramid Scale',
        type: 'slider',
        min: 0.1,
        max: 0.9,
        step: 0.1,
        default: 0.5
      },
      {
        id: 'levels',
        name: 'Pyramid Levels',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 3
      },
      {
        id: 'winsize',
        name: 'Window Size',
        type: 'slider',
        min: 5,
        max: 25,
        step: 2,
        default: 15
      }
    ]
  },
  {
    id: 'meanshift_tracker',
    name: 'MeanShift Tracker',
    description: 'Color-based object tracking',
    category: 'video',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'maxIter',
        name: 'Max Iterations',
        type: 'slider',
        min: 1,
        max: 100,
        step: 1,
        default: 10
      }
    ]
  },
  {
    id: 'camshift_tracker',
    name: 'CAMShift Tracker',
    description: 'Adaptive mean shift tracking',
    category: 'video',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'mosse_tracker',
    name: 'MOSSE Tracker',
    description: 'Fast correlation filter tracking',
    category: 'video',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'learningRate',
        name: 'Learning Rate',
        type: 'slider',
        min: 0.01,
        max: 0.5,
        step: 0.01,
        default: 0.125
      }
    ]
  },
  {
    id: 'csrt_tracker',
    name: 'CSRT Tracker',
    description: 'Discriminative correlation filter with spatial reliability',
    category: 'video',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'bg_subtractor_mog2',
    name: 'Background Subtractor MOG2',
    description: 'Gaussian mixture-based background/foreground segmentation',
    category: 'video',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'history',
        name: 'History',
        type: 'slider',
        min: 1,
        max: 500,
        step: 10,
        default: 500
      },
      {
        id: 'varThreshold',
        name: 'Variance Threshold',
        type: 'slider',
        min: 4,
        max: 100,
        step: 1,
        default: 16
      },
      {
        id: 'detectShadows',
        name: 'Detect Shadows',
        type: 'checkbox',
        default: true
      }
    ]
  },
  {
    id: 'bg_subtractor_knn',
    name: 'Background Subtractor KNN',
    description: 'K-nearest neighbors background subtraction',
    category: 'video',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'history',
        name: 'History',
        type: 'slider',
        min: 1,
        max: 500,
        step: 10,
        default: 500
      },
      {
        id: 'dist2Threshold',
        name: 'Distance Threshold',
        type: 'slider',
        min: 100,
        max: 1000,
        step: 50,
        default: 400
      }
    ]
  },

  // ========================================
  // CATEGORY 12: CAMERA CALIBRATION
  // ========================================

  {
    id: 'calibrate_camera',
    name: 'Calibrate Camera',
    description: 'Camera calibration using chessboard',
    category: 'calibration',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'patternWidth',
        name: 'Pattern Width',
        type: 'slider',
        min: 3,
        max: 20,
        step: 1,
        default: 9
      },
      {
        id: 'patternHeight',
        name: 'Pattern Height',
        type: 'slider',
        min: 3,
        max: 20,
        step: 1,
        default: 6
      },
      {
        id: 'squareSize',
        name: 'Square Size (mm)',
        type: 'slider',
        min: 1,
        max: 100,
        step: 1,
        default: 25
      }
    ]
  },
  {
    id: 'fisheye_calibration',
    name: 'Fisheye Calibration',
    description: 'Wide-angle lens calibration',
    category: 'calibration',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'solve_pnp',
    name: 'Solve PnP',
    description: 'Find object pose from 3D-2D correspondences',
    category: 'calibration',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'method',
        name: 'Method',
        type: 'select',
        options: ['Iterative', 'P3P', 'EPNP', 'DLS'],
        default: 'Iterative'
      }
    ]
  },
  {
    id: 'stereo_calibration',
    name: 'Stereo Calibration',
    description: 'Calibrate stereo camera pair',
    category: 'calibration',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'stereo_rectification',
    name: 'Stereo Rectification',
    description: 'Align stereo images for disparity computation',
    category: 'calibration',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'compute_disparity',
    name: 'Compute Disparity',
    description: 'Generate depth map from stereo pair',
    category: 'calibration',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'blockSize',
        name: 'Block Size',
        type: 'slider',
        min: 5,
        max: 255,
        step: 2,
        default: 15
      },
      {
        id: 'numDisparities',
        name: 'Num Disparities',
        type: 'slider',
        min: 16,
        max: 256,
        step: 16,
        default: 64
      }
    ]
  },
  {
    id: 'find_homography',
    name: 'Find Homography',
    description: 'Estimate planar transformation from point correspondences',
    category: 'calibration',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'method',
        name: 'Method',
        type: 'select',
        options: ['RANSAC', 'LMeDS', 'RHO'],
        default: 'RANSAC'
      },
      {
        id: 'ransacThreshold',
        name: 'RANSAC Threshold',
        type: 'slider',
        min: 1,
        max: 10,
        step: 0.5,
        default: 3.0
      }
    ]
  },

  // ========================================
  // CATEGORY 13: MACHINE LEARNING
  // ========================================

  {
    id: 'svm_classifier',
    name: 'SVM Classifier',
    description: 'Support Vector Machine for classification',
    category: 'ml',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'kernelType',
        name: 'Kernel Type',
        type: 'select',
        options: ['Linear', 'RBF', 'Poly', 'Sigmoid'],
        default: 'RBF'
      },
      {
        id: 'C',
        name: 'C Parameter',
        type: 'slider',
        min: 0.1,
        max: 100,
        step: 0.1,
        default: 1.0
      },
      {
        id: 'gamma',
        name: 'Gamma',
        type: 'slider',
        min: 0.001,
        max: 10,
        step: 0.01,
        default: 1.0
      }
    ]
  },
  {
    id: 'decision_tree',
    name: 'Decision Tree',
    description: 'Decision tree classifier',
    category: 'ml',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'maxDepth',
        name: 'Max Depth',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 5
      }
    ]
  },
  {
    id: 'random_forest',
    name: 'Random Forest',
    description: 'Random forest ensemble classifier',
    category: 'ml',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'numTrees',
        name: 'Number of Trees',
        type: 'slider',
        min: 1,
        max: 200,
        step: 10,
        default: 100
      },
      {
        id: 'maxDepth',
        name: 'Max Depth',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 5
      }
    ]
  },
  {
    id: 'knn',
    name: 'K-Nearest Neighbors',
    description: 'Instance-based learning classifier',
    category: 'ml',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'k',
        name: 'K Value',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 5
      },
      {
        id: 'distanceMetric',
        name: 'Distance Metric',
        type: 'select',
        options: ['L2', 'L1'],
        default: 'L2'
      }
    ]
  },
  {
    id: 'neural_network',
    name: 'Neural Network (MLP)',
    description: 'Multi-layer perceptron neural network',
    category: 'ml',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'learningRate',
        name: 'Learning Rate',
        type: 'slider',
        min: 0.001,
        max: 1.0,
        step: 0.01,
        default: 0.1
      },
      {
        id: 'iterations',
        name: 'Iterations',
        type: 'slider',
        min: 100,
        max: 10000,
        step: 100,
        default: 1000
      }
    ]
  },
  {
    id: 'kmeans',
    name: 'K-Means Clustering',
    description: 'Unsupervised clustering algorithm',
    category: 'ml',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'k',
        name: 'Number of Clusters',
        type: 'slider',
        min: 2,
        max: 20,
        step: 1,
        default: 3
      },
      {
        id: 'maxIterations',
        name: 'Max Iterations',
        type: 'slider',
        min: 10,
        max: 1000,
        step: 10,
        default: 100
      },
      {
        id: 'attempts',
        name: 'Attempts',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 3
      }
    ]
  },

  // ========================================
  // CATEGORY 14: COMPUTATIONAL PHOTOGRAPHY
  // ========================================

  {
    id: 'merge_debevec',
    name: 'Merge Debevec (HDR)',
    description: 'Create HDR image from exposure stack',
    category: 'photo',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'tonemap_drago',
    name: 'Tonemap Drago',
    description: 'HDR to LDR using Drago tone mapping',
    category: 'photo',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'gamma',
        name: 'Gamma',
        type: 'slider',
        min: 0.1,
        max: 3.0,
        step: 0.1,
        default: 1.0
      },
      {
        id: 'saturation',
        name: 'Saturation',
        type: 'slider',
        min: 0.0,
        max: 2.0,
        step: 0.1,
        default: 1.0
      }
    ]
  },
  {
    id: 'tonemap_reinhard',
    name: 'Tonemap Reinhard',
    description: 'Local tone mapping operator',
    category: 'photo',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'intensity',
        name: 'Intensity',
        type: 'slider',
        min: -8,
        max: 8,
        step: 0.1,
        default: 0.0
      },
      {
        id: 'lightAdapt',
        name: 'Light Adapt',
        type: 'slider',
        min: 0.0,
        max: 1.0,
        step: 0.1,
        default: 1.0
      }
    ]
  },
  {
    id: 'fast_nl_means',
    name: 'Fast NL Means Denoising',
    description: 'Fast non-local means denoising',
    category: 'photo',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'h',
        name: 'H',
        type: 'slider',
        min: 1,
        max: 30,
        step: 1,
        default: 10
      },
      {
        id: 'templateWindowSize',
        name: 'Template Window',
        type: 'slider',
        min: 7,
        max: 21,
        step: 2,
        default: 7
      },
      {
        id: 'searchWindowSize',
        name: 'Search Window',
        type: 'slider',
        min: 21,
        max: 35,
        step: 2,
        default: 21
      }
    ]
  },
  {
    id: 'inpaint',
    name: 'Inpaint',
    description: 'Image restoration and object removal',
    category: 'photo',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'method',
        name: 'Method',
        type: 'select',
        options: ['Navier-Stokes', 'Telea'],
        default: 'Telea'
      },
      {
        id: 'radius',
        name: 'Radius',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 5
      }
    ]
  },
  {
    id: 'super_resolution',
    name: 'Super Resolution',
    description: 'Upscale images with enhanced detail',
    category: 'photo',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'scale',
        name: 'Scale',
        type: 'select',
        options: ['2x', '3x', '4x'],
        default: '2x'
      }
    ]
  },

  // ========================================
  // CATEGORY 15: IMAGE STITCHING & PANORAMA
  // ========================================

  {
    id: 'panorama_stitcher',
    name: 'Panorama Stitcher',
    description: 'Automatic panorama creation from overlapping images',
    category: 'stitching',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'confidenceThreshold',
        name: 'Confidence Threshold',
        type: 'slider',
        min: 0.1,
        max: 2.0,
        step: 0.1,
        default: 1.0
      }
    ]
  },
  {
    id: 'feather_blender',
    name: 'Feather Blender',
    description: 'Simple alpha blending for panoramas',
    category: 'stitching',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'sharpness',
        name: 'Sharpness',
        type: 'slider',
        min: 0.01,
        max: 1.0,
        step: 0.01,
        default: 0.02
      }
    ]
  },
  {
    id: 'multiband_blender',
    name: 'Multi-band Blender',
    description: 'Pyramid blending for seamless panoramas',
    category: 'stitching',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'numBands',
        name: 'Number of Bands',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 5
      }
    ]
  },

  // ========================================
  // CATEGORY 16: DRAWING & ANNOTATION
  // ========================================

  {
    id: 'draw_line',
    name: 'Draw Line',
    description: 'Draw a line on the image',
    category: 'drawing',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'thickness',
        name: 'Line Thickness',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 2
      },
      {
        id: 'color',
        name: 'Color',
        type: 'color',
        default: '#ff0000'
      }
    ]
  },
  {
    id: 'draw_rectangle',
    name: 'Draw Rectangle',
    description: 'Draw a rectangle on the image',
    category: 'drawing',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'thickness',
        name: 'Line Thickness',
        type: 'slider',
        min: -1,
        max: 20,
        step: 1,
        default: 2,
        description: '-1 for filled rectangle'
      },
      {
        id: 'color',
        name: 'Color',
        type: 'color',
        default: '#00ff00'
      }
    ]
  },
  {
    id: 'draw_circle',
    name: 'Draw Circle',
    description: 'Draw a circle on the image',
    category: 'drawing',
    implemented: true,
    gpuAccelerated: false,
    params: [
      {
        id: 'radius',
        name: 'Radius',
        type: 'slider',
        min: 5,
        max: 200,
        step: 5,
        default: 50
      },
      {
        id: 'thickness',
        name: 'Line Thickness',
        type: 'slider',
        min: -1,
        max: 20,
        step: 1,
        default: 2
      },
      {
        id: 'color',
        name: 'Color',
        type: 'color',
        default: '#0000ff'
      }
    ]
  },
  {
    id: 'draw_ellipse',
    name: 'Draw Ellipse',
    description: 'Draw an ellipse on the image',
    category: 'drawing',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'angle',
        name: 'Rotation Angle',
        type: 'slider',
        min: 0,
        max: 360,
        step: 5,
        default: 0
      },
      {
        id: 'thickness',
        name: 'Line Thickness',
        type: 'slider',
        min: -1,
        max: 20,
        step: 1,
        default: 2
      },
      {
        id: 'color',
        name: 'Color',
        type: 'color',
        default: '#ff00ff'
      }
    ]
  },
  {
    id: 'draw_polylines',
    name: 'Draw Polylines',
    description: 'Draw polygon/polylines on the image',
    category: 'drawing',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'closed',
        name: 'Closed',
        type: 'checkbox',
        default: false
      },
      {
        id: 'thickness',
        name: 'Line Thickness',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 2
      },
      {
        id: 'color',
        name: 'Color',
        type: 'color',
        default: '#ffff00'
      }
    ]
  },
  {
    id: 'put_text',
    name: 'Put Text',
    description: 'Render text on the image',
    category: 'drawing',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'text',
        name: 'Text',
        type: 'text',
        default: 'Hello OpenCV'
      },
      {
        id: 'fontScale',
        name: 'Font Scale',
        type: 'slider',
        min: 0.5,
        max: 5.0,
        step: 0.1,
        default: 1.0
      },
      {
        id: 'thickness',
        name: 'Thickness',
        type: 'slider',
        min: 1,
        max: 10,
        step: 1,
        default: 2
      },
      {
        id: 'color',
        name: 'Color',
        type: 'color',
        default: '#ffffff'
      }
    ]
  },

  // ========================================
  // CATEGORY 17: DEEP NEURAL NETWORKS
  // ========================================

  {
    id: 'load_network',
    name: 'Load Neural Network',
    description: 'Load a pre-trained DNN model',
    category: 'dnn',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'framework',
        name: 'Framework',
        type: 'select',
        options: ['TensorFlow', 'PyTorch', 'ONNX', 'Caffe'],
        default: 'ONNX'
      }
    ]
  },
  {
    id: 'blob_from_image',
    name: 'Blob from Image',
    description: 'Prepare image for DNN inference',
    category: 'dnn',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'scaleFactor',
        name: 'Scale Factor',
        type: 'slider',
        min: 0.0,
        max: 1.0,
        step: 0.01,
        default: 0.00392
      },
      {
        id: 'swapRB',
        name: 'Swap RB',
        type: 'checkbox',
        default: true
      }
    ]
  },

  // ========================================
  // CATEGORY 18: SHAPE ANALYSIS
  // ========================================

  {
    id: 'min_enclosing_circle',
    name: 'Min Enclosing Circle',
    description: 'Find smallest circle enclosing contour',
    category: 'shape',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'convex_hull',
    name: 'Convex Hull',
    description: 'Compute convex boundary of contour',
    category: 'shape',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'clockwise',
        name: 'Clockwise',
        type: 'checkbox',
        default: false
      }
    ]
  },
  {
    id: 'hu_moments',
    name: 'Hu Moments',
    description: 'Rotation-invariant shape descriptors',
    category: 'shape',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
  {
    id: 'match_shapes',
    name: 'Match Shapes',
    description: 'Compare shape similarity',
    category: 'shape',
    implemented: false,
    gpuAccelerated: false,
    params: [
      {
        id: 'method',
        name: 'Method',
        type: 'select',
        options: ['I1', 'I2', 'I3'],
        default: 'I1'
      }
    ]
  }
];

// Helper function to get demos by category
export const getDemosByCategory = (categoryId) => {
  return demos.filter(demo => demo.category === categoryId);
};

// Helper function to get demo by ID
export const getDemoById = (demoId) => {
  return demos.find(demo => demo.id === demoId);
};

// Get category info
export const getCategoryById = (categoryId) => {
  return categories.find(cat => cat.id === categoryId);
};

// Get default params for a demo
export const getDefaultParams = (demoId) => {
  const demo = getDemoById(demoId);
  if (!demo) return {};

  const defaultParams = {};
  demo.params.forEach(param => {
    defaultParams[param.id] = param.default;
  });
  return defaultParams;
};

// Get statistics
export const getStats = () => {
  const total = demos.length;
  const implemented = demos.filter(d => d.implemented).length;
  const gpuAccelerated = demos.filter(d => d.gpuAccelerated).length;

  return {
    total,
    implemented,
    notImplemented: total - implemented,
    gpuAccelerated,
    completion: ((implemented / total) * 100).toFixed(1)
  };
};
