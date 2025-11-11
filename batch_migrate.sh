#!/bin/bash
# Batch migration script to add backend selection pattern to all remaining WASM operations
# This script creates a migration summary and prepares for systematic updates

echo "=== Backend Selection Migration Status ==="
echo ""
echo "Completed (23/102 operations):"
echo "- basic/edge.rs (4): canny, sobel, scharr, laplacian"
echo "- basic/filtering.rs (9): gaussian_blur, blur, box_blur, median_blur, bilateral_filter, guided_filter, gabor_filter, nlm_denoising, anisotropic_diffusion, fast_nl_means, filter2d"
echo "- basic/threshold.rs (1): adaptive_threshold"
echo "- imgproc/morphology.rs (9): erode, dilate, opening, closing, gradient, tophat, blackhat, tophat_alt, blackhat_alt"
echo ""
echo "Remaining (79/102 operations):"
echo ""

# Count remaining operations per file
for file in src/wasm/imgproc/geometric.rs src/wasm/imgproc/color.rs src/wasm/imgproc/histogram.rs \
src/wasm/imgproc/contour.rs src/wasm/imgproc/drawing.rs \
src/wasm/arithmetic/ops.rs src/wasm/comparison/bitwise.rs \
src/wasm/features/*.rs src/wasm/ml/*.rs src/wasm/video/*.rs \
src/wasm/calib3d/*.rs src/wasm/dnn/*.rs src/wasm/segmentation/*.rs src/wasm/misc/*.rs; do
    if [ -f "$file" ]; then
        count=$(grep -c "pub async fn.*_wasm" "$file" 2>/dev/null || echo "0")
        if [ "$count" -gt 0 ]; then
            echo "- $file: $count operations"
        fi
    fi
done
