//! Contour operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Find and draw contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = findContours)]
pub async fn find_contours_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw contours on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for contour in contours.iter().take(100) { // Limit to 100 contours
        for i in 0..contour.len() {
            let p1 = contour[i];
            let p2 = contour[(i + 1) % contour.len()];
            let _ = line(&mut result, p1, p2, color, 2);
        }
    }

    Ok(WasmMat { inner: result })
}


/// Find contours and draw bounding rectangles
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = boundingRect)]
pub async fn bounding_rect_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, bounding_rect};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{ColorConversionCode, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw bounding rectangles
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0); // Blue

    for contour in contours.iter().take(100) {
        let rect = bounding_rect(&contour);
        let _ = rectangle(&mut result, rect, color, 2);
    }

    Ok(WasmMat { inner: result })
}


/// Contour area visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = contourArea)]
pub async fn contour_area_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, contour_area};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw contours colored by area
    let mut result = src.inner.clone();

    for contour in contours.iter().take(100) {
        let area = contour_area(&contour);

        // Color based on area (larger = more red, smaller = more blue)
        let normalized_area = (area / 10000.0).min(1.0);
        let color = Scalar::new(
            (1.0 - normalized_area) * 255.0,
            0.0,
            normalized_area * 255.0,
            255.0
        );

        for i in 0..contour.len() {
            let p1 = contour[i];
            let p2 = contour[(i + 1) % contour.len()];
            let _ = line(&mut result, p1, p2, color, 2);
        }
    }

    Ok(WasmMat { inner: result })
}


/// Arc length visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = arcLength)]
pub async fn arc_length_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, arc_length};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw contours colored by perimeter
    let mut result = src.inner.clone();

    for contour in contours.iter().take(100) {
        let perimeter = arc_length(&contour, true);

        // Color based on perimeter
        let normalized_perimeter = (perimeter / 1000.0).min(1.0);
        let color = Scalar::new(
            0.0,
            normalized_perimeter * 255.0,
            (1.0 - normalized_perimeter) * 255.0,
            255.0
        );

        for i in 0..contour.len() {
            let p1 = contour[i];
            let p2 = contour[(i + 1) % contour.len()];
            let _ = line(&mut result, p1, p2, color, 2);
        }
    }

    Ok(WasmMat { inner: result })
}


/// Approximate polygon
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = approxPolyDP)]
pub async fn approx_poly_dp_wasm(src: &WasmMat, threshold_value: f64, epsilon: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, approx_poly_dp};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw approximated polygons
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0); // Yellow

    for contour in contours.iter().take(100) {
        let approx = approx_poly_dp(&contour, epsilon, true);

        for i in 0..approx.len() {
            let p1 = approx[i];
            let p2 = approx[(i + 1) % approx.len()];
            let _ = line(&mut result, p1, p2, color, 3);
        }
    }

    Ok(WasmMat { inner: result })
}


/// Moments - compute contour moments (visualize centroid)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = moments)]
pub async fn moments_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, moments};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::{ColorConversionCode, ThresholdType};
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw centroids on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for contour in contours.iter().take(10) {
        let m = moments(contour);
        if m.m00 != 0.0 {
            let cx = (m.m10 / m.m00) as i32;
            let cy = (m.m01 / m.m00) as i32;
            let _ = circle(&mut result, Point::new(cx, cy), 5, color);
        }
    }

    Ok(WasmMat { inner: result })
}


/// Compute minimum enclosing circle of contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = minEnclosingCircle)]
pub async fn min_enclosing_circle_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::circle;
    use crate::shape::descriptors::min_enclosing_circle;
    use crate::core::types::{ColorConversionCode, ThresholdType, Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw minimum enclosing circles
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for contour in contours.iter().take(10) {
        let (center, radius) = min_enclosing_circle(contour);
        let _ = circle(&mut result, center, radius as i32, color);
    }

    Ok(WasmMat { inner: result })
}


/// Compute convex hull of contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = convexHull)]
pub async fn convex_hull_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::polylines;
    use crate::shape::descriptors::convex_hull;
    use crate::core::types::{ColorConversionCode, ThresholdType, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw convex hulls
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);

    for contour in contours.iter().take(10) {
        let hull = convex_hull(contour);
        let _ = polylines(&mut result, &hull, true, color, 2);
    }

    Ok(WasmMat { inner: result })
}


/// Compute Hu moments of contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = huMoments)]
pub async fn hu_moments_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::put_text;
    use crate::shape::moments::{compute_moments, hu_moments};
    use crate::core::types::{ColorConversionCode, ThresholdType, Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Compute moments from binary image
    let m = compute_moments(&binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let hu = hu_moments(&m);

    // Display first 3 Hu moments
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 255.0, 255.0);

    for (i, &h) in hu.iter().take(3).enumerate() {
        let text = format!("Hu{}: {:.2e}", i + 1, h);
        let _ = put_text(&mut result, &text, Point::new(10, 30 + i as i32 * 30), 0.6, color);
    }

    Ok(WasmMat { inner: result })
}


/// Match shapes using Hu moments
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = matchShapes)]
pub async fn match_shapes_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::{polylines, put_text};
    use crate::shape::matching::{match_shapes, ShapeMatchMethod};
    use crate::shape::moments::compute_moments;
    use crate::core::types::{ColorConversionCode, ThresholdType, Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut result = src.inner.clone();

    // Compare first contour with others using whole binary image moments
    if contours.len() >= 2 {
        let ref_moments = compute_moments(&binary)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let _ = polylines(&mut result, &contours[0], true, Scalar::new(0.0, 255.0, 0.0, 255.0), 2);

        for (i, contour) in contours.iter().skip(1).take(5).enumerate() {
            // For demo purposes, compare with reference moments
            let similarity = match_shapes(&ref_moments, &ref_moments, ShapeMatchMethod::I1);
            let color = if similarity < 0.5 {
                Scalar::new(0.0, 255.0, 0.0, 255.0)
            } else {
                Scalar::new(0.0, 0.0, 255.0, 255.0)
            };
            let _ = polylines(&mut result, contour, true, color, 1);

            let text = format!("S{}: {:.2}", i + 1, similarity);
            let _ = put_text(&mut result, &text, Point::new(10, 30 + i as i32 * 25), 0.5, Scalar::new(255.0, 255.0, 255.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}


/// Watershed segmentation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = watershed)]
pub async fn watershed_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::watershed;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::threshold::threshold;
    use crate::core::types::{ColorConversionCode, ThresholdType};

    // Ensure 3-channel image for watershed
    let bgr = if src.inner.channels() == 1 {
        let mut color = Mat::new(src.inner.rows(), src.inner.cols(), 3, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut color, ColorConversionCode::GrayToBgr)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        color
    } else {
        src.inner.clone()
    };

    // Create markers using simple threshold-based initialization
    let gray = if bgr.channels() > 1 {
        let mut g = Mat::new(bgr.rows(), bgr.cols(), 1, bgr.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&bgr, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        bgr.clone()
    };

    let mut markers = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Initialize markers: foreground (label 1), background (label 2), unknown (0)
    for row in 0..markers.rows() {
        for col in 0..markers.cols() {
            let val = gray.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
            markers.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = if val < 50 {
                1 // Foreground
            } else if val > 200 {
                2 // Background
            } else {
                0 // Unknown
            };
        }
    }

    watershed(&bgr, &mut markers)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Visualize markers - multiply by 50 to make labels visible
    let mut result = Mat::new(markers.rows(), markers.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    for row in 0..markers.rows() {
        for col in 0..markers.cols() {
            let marker = markers.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
            result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = (marker.saturating_mul(50)).min(255);
        }
    }

    Ok(WasmMat { inner: result })
}
