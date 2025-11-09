/**
 * Demo Registry - MECE categorization of all OpenCV demos
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
    name: '🎨 Image Filtering',
    description: 'Smoothing and noise reduction filters',
    priority: 0
  },
  {
    id: 'edges',
    name: '📐 Edge Detection',
    description: 'Edge and derivative operators',
    priority: 0
  },
  {
    id: 'transform',
    name: '🔄 Geometric Transform',
    description: 'Resize, rotate, warp operations',
    priority: 0
  },
  {
    id: 'color',
    name: '🌈 Color & Threshold',
    description: 'Color space and thresholding',
    priority: 0
  },
  {
    id: 'histogram',
    name: '📊 Histogram Operations',
    description: 'Histogram analysis and equalization',
    priority: 1
  },
  {
    id: 'morphology',
    name: '🔲 Morphology',
    description: 'Erode, dilate, and morphological operations',
    priority: 1
  },
  {
    id: 'contours',
    name: '🎯 Contours',
    description: 'Contour detection and analysis',
    priority: 0
  },
  {
    id: 'features',
    name: '🎯 Feature Detection',
    description: 'Keypoints, descriptors, and matching',
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
    description: 'HOG, cascade, ArUco, QR codes',
    priority: 1
  },
  {
    id: 'video',
    name: '🎥 Video Analysis',
    description: 'Tracking and optical flow',
    priority: 2
  },
  {
    id: 'calibration',
    name: '📷 Camera Calibration',
    description: 'Calibration and pose estimation',
    priority: 2
  },
  {
    id: 'ml',
    name: '🤖 Machine Learning',
    description: 'SVM, Random Forest, Neural Networks',
    priority: 2
  },
  {
    id: 'photo',
    name: '📸 Computational Photography',
    description: 'HDR, denoising, inpainting',
    priority: 2
  },
  {
    id: 'stitching',
    name: '🌄 Image Stitching',
    description: 'Panorama creation and blending',
    priority: 3
  },
  {
    id: 'drawing',
    name: '✏️ Drawing',
    description: 'Shapes, lines, and text',
    priority: 1
  }
];

export const demos = [
  // ========== FILTERS ==========
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
    implemented: false,
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
        options: [3, 5, 7, 9, 11],
        default: 5
      }
    ]
  },
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

  // ========== EDGES ==========
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
      }
    ]
  },
  {
    id: 'sobel',
    name: 'Sobel Operator',
    description: 'Gradient-based edge detection',
    category: 'edges',
    implemented: false,
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

  // ========== TRANSFORM ==========
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
        options: ['Nearest', 'Linear', 'Cubic'],
        default: 'Linear'
      }
    ]
  },
  {
    id: 'rotate',
    name: 'Rotate',
    description: 'Rotate image by 90, 180, or 270 degrees',
    category: 'transform',
    implemented: false,
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
  {
    id: 'flip',
    name: 'Flip',
    description: 'Flip image horizontally or vertically',
    category: 'transform',
    implemented: false,
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

  // ========== COLOR & THRESHOLD ==========
  {
    id: 'cvt_color_gray',
    name: 'Convert to Grayscale',
    description: 'Convert color image to grayscale',
    category: 'color',
    implemented: false,
    gpuAccelerated: false,
    params: []
  },
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

  // ========== MORPHOLOGY ==========
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
      }
    ]
  },

  // ========== DRAWING ==========
  {
    id: 'draw_line',
    name: 'Draw Line',
    description: 'Draw a line on the image',
    category: 'drawing',
    implemented: false,
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
    implemented: false,
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
    implemented: false,
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
