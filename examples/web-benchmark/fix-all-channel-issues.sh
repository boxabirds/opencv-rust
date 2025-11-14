#!/bin/bash
# Fix all "3 channels" errors by handling RGBA input properly

echo "Fixing all BgrToGray conversions to handle RGBA..."

# Pattern to find and replace
# FROM: cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
# TO:
#       let conversion_code = if src.inner.channels() == 4 {
#           ColorConversionCode::RgbaToGray
#       } else {
#           ColorConversionCode::BgrToGray
#       };
#       cvt_color(&src.inner, &mut g, conversion_code)

# But this is complex for sed/awk, so let's identify files and fix manually

find ../../src/wasm -name "*.rs" -exec grep -l "ColorConversionCode::BgrToGray" {} \; | sort -u
