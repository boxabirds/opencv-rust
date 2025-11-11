#!/usr/bin/env python3
"""
Script to systematically migrate all WASM operations to use backend selection pattern.
This automates the tedious process of updating each operation.
"""

import re
import sys
from pathlib import Path

# List of GPU operations available (from src/gpu/ops/)
GPU_OPS = {
    'canny', 'sobel', 'scharr', 'laplacian',  # edge
    'gaussian_blur', 'box_blur', 'median_blur', 'bilateral_filter', 'filter2d',  # filtering
    'erode', 'dilate',  # morphology
    'resize', 'flip', 'rotate', 'warp_affine', 'warp_perspective',  # geometric
    'threshold', 'adaptive_threshold',  # threshold
    'rgb_to_gray', 'rgb_to_hsv', 'rgb_to_lab', 'rgb_to_ycrcb',  # color
    'hsv_to_rgb', 'lab_to_rgb', 'ycrcb_to_rgb',
    'add', 'subtract', 'multiply', 'add_weighted', 'absdiff',  # arithmetic
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',  # bitwise
    'min', 'max', 'convert_scale', 'normalize', 'in_range',  # comparison
    'pyrdown', 'pyrup', 'remap', 'lut',  # pyramid/transform
    'split', 'merge',  # channels
    'exp', 'log', 'sqrt', 'pow',  # math
    'equalize_hist', 'integral_image', 'distance_transform',  # histogram/transform
    'gradient_magnitude'  # gradients
}

def has_gpu_support(op_name):
    """Check if operation likely has GPU support based on common patterns"""
    base_name = op_name.replace('_wasm', '').replace('_async', '')
    return base_name in GPU_OPS

def get_gpu_function_name(op_name):
    """Convert WASM function name to GPU function name"""
    base = op_name.replace('_wasm', '')
    return f"{base}_gpu_async"

# Count found
print("GPU operations available:", len(GPU_OPS))
print("Sample GPU ops:", sorted(list(GPU_OPS))[:10])
