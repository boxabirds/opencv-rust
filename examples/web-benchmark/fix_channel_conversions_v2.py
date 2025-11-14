#!/usr/bin/env python3
"""
Fix all BgrToGray conversions to handle RGBA input (4 channels)
Version 2: Handles fully qualified paths properly
"""

import re
import sys

files = [
    "../../src/wasm/basic/edge.rs",
    "../../src/wasm/basic/filtering.rs",
    "../../src/wasm/basic/threshold.rs",
    "../../src/wasm/calib3d/camera.rs",
    "../../src/wasm/features/detection.rs",
    "../../src/wasm/features/object.rs",
    "../../src/wasm/imgproc/contour.rs",
    "../../src/wasm/imgproc/histogram.rs",
    "../../src/wasm/misc/various.rs",
    "../../src/wasm/segmentation/cluster.rs",
    "../../src/wasm/video/tracking.rs",
]

# More comprehensive pattern that handles optional module prefix
# Matches:
#   cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
#   crate::imgproc::color::cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
pattern = r'([\w:]*)(cvt_color)\(&([a-z_\.]+), &mut g, ColorConversionCode::BgrToGray\)'

def replacement(match):
    prefix = match.group(1)  # e.g., "" or "crate::imgproc::color::"
    func = match.group(2)     # "cvt_color"
    var = match.group(3)      # e.g., "src.inner" or "gray"

    return f'''// Use correct color conversion based on number of channels
        let conversion_code = if {var}.channels() == 4 {{
            ColorConversionCode::RgbaToGray
        }} else {{
            ColorConversionCode::BgrToGray
        }};
        {prefix}{func}(&{var}, &mut g, conversion_code)'''

for file_path in files:
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        original = content
        content = re.sub(pattern, replacement, content)

        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✓ Fixed: {file_path}")
        else:
            print(f"- No changes: {file_path}")

    except Exception as e:
        print(f"✗ Error in {file_path}: {e}", file=sys.stderr)

print("\nDone!")
