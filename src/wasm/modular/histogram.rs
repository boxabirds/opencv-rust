//! Histogram operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Equalize histogram for contrast enhancement
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = equalizeHistogram)]
pub async fn equalize_histogram_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::equalize_hist;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    equalize_hist(&gray, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Calculate histogram (returns visual representation)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = calcHistogram)]
pub async fn calc_histogram_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::calc_hist;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{ColorConversionCode, Rect, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    // Calculate histogram
    let hist = calc_hist(&gray, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create visualization image (256x256)
    let hist_img_size = 256;
    let mut hist_img = Mat::new(hist_img_size, hist_img_size, 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Fill white background
    for row in 0..hist_img_size {
        for col in 0..hist_img_size {
            let pixel = hist_img.at_mut(row, col)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }

    // Find max value for scaling
    let max_val = hist.iter().cloned().fold(0.0f32, f32::max);

    // Draw histogram bars
    let bin_width = hist_img_size / 256;
    for i in 0..256 {
        let bin_height = if max_val > 0.0 {
            ((hist[i] / max_val) * hist_img_size as f32) as i32
        } else {
            0
        };

        if bin_height > 0 {
            let rect = Rect::new(
                i as i32 * bin_width as i32,
                hist_img_size as i32 - bin_height,
                bin_width as i32,
                bin_height,
            );
            let _ = rectangle(&mut hist_img, rect, Scalar::new(0.0, 0.0, 0.0, 255.0), -1);
        }
    }

    Ok(WasmMat { inner: hist_img })
}


/// Normalize histogram (returns visualization)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = normalizeHistogram)]
pub async fn normalize_histogram_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::{calc_hist, normalize_hist};
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    // Calculate and normalize histogram
    let mut hist = calc_hist(&gray, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    normalize_hist(&mut hist, 0.0, 1.0);

    // Create visualization
    let hist_img_size = 256;
    let mut hist_img = Mat::new(hist_img_size, hist_img_size, 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Fill white background
    for row in 0..hist_img_size {
        for col in 0..hist_img_size {
            let pixel = hist_img.at_mut(row, col)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }

    // Draw normalized histogram bars
    let bin_width = hist_img_size / 256;
    for i in 0..256 {
        let bin_height = (hist[i] * hist_img_size as f32) as i32;
        if bin_height > 0 {
            let rect = Rect::new(
                i as i32 * bin_width as i32,
                hist_img_size as i32 - bin_height,
                bin_width as i32,
                bin_height,
            );
            let _ = rectangle(&mut hist_img, rect, Scalar::new(0.0, 255.0, 0.0, 255.0), -1);
        }
    }

    Ok(WasmMat { inner: hist_img })
}


/// Compare histograms (returns similarity score)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = compareHistograms)]
pub async fn compare_histograms_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<f64, JsValue> {
    use crate::imgproc::histogram::{calc_hist, compare_hist, HistCompMethod};
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Convert both to grayscale
    let gray1 = if src1.inner.channels() > 1 {
        let mut g = Mat::new(src1.inner.rows(), src1.inner.cols(), 1, src1.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src1.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src1.inner.clone()
    };

    let gray2 = if src2.inner.channels() > 1 {
        let mut g = Mat::new(src2.inner.rows(), src2.inner.cols(), 1, src2.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src2.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src2.inner.clone()
    };

    // Calculate histograms
    let hist1 = calc_hist(&gray1, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let hist2 = calc_hist(&gray2, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Compare using correlation method
    let similarity = compare_hist(&hist1, &hist2, HistCompMethod::Correlation)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(similarity)
}


/// Back projection
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = backProjection)]
pub async fn back_projection_wasm(src: &WasmMat, model: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::{calc_back_project, calc_hist};
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Convert both to grayscale
    let gray_src = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let gray_model = if model.inner.channels() > 1 {
        let mut g = Mat::new(model.inner.rows(), model.inner.cols(), 1, model.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&model.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        model.inner.clone()
    };

    // Calculate histogram of model image
    let model_hist = calc_hist(&gray_model, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut dst = Mat::new(gray_src.rows(), gray_src.cols(), 1, gray_src.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    calc_back_project(&gray_src, &model_hist, (0.0, 256.0), &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}
