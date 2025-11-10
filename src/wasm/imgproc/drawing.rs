//! Drawing operations

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::wasm::WasmMat;

/// Draw a line on the image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawLine)]
pub async fn draw_line_wasm(
    src: &WasmMat,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    r: u8,
    g: u8,
    b: u8,
    thickness: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::line;
    use crate::core::types::{Point, Scalar};

    let mut img = src.inner.clone();
    let pt1 = Point::new(x1, y1);
    let pt2 = Point::new(x2, y2);
    let color = Scalar::new(b as f64, g as f64, r as f64, 255.0);

    line(&mut img, pt1, pt2, color, thickness)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: img })
}

/// Draw a rectangle on the image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawRectangle)]
pub async fn draw_rectangle_wasm(
    src: &WasmMat,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    r: u8,
    g: u8,
    b: u8,
    thickness: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    let mut img = src.inner.clone();
    let rect = Rect::new(x, y, width, height);
    let color = Scalar::new(b as f64, g as f64, r as f64, 255.0);

    rectangle(&mut img, rect, color, thickness)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: img })
}

/// Draw a circle on the image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawCircle)]
pub async fn draw_circle_wasm(
    src: &WasmMat,
    center_x: i32,
    center_y: i32,
    radius: i32,
    r: u8,
    g: u8,
    b: u8,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    let mut img = src.inner.clone();
    let center = Point::new(center_x, center_y);
    let color = Scalar::new(b as f64, g as f64, r as f64, 255.0);

    circle(&mut img, center, radius, color)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: img })
}

/// Draw ellipse on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawEllipse)]
pub async fn draw_ellipse_wasm(src: &WasmMat, cx: i32, cy: i32, width: i32, height: i32, angle: f64, thickness: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::ellipse;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();
    let center = Point::new(cx, cy);
    let axes = (width / 2, height / 2);
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    ellipse(&mut result, center, axes, angle, 0.0, 360.0, color)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: result })
}

/// Draw polylines on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawPolylines)]
pub async fn draw_polylines_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::polylines;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);

    // Create a sample polygon (diamond shape)
    let w = result.cols() as i32;
    let h = result.rows() as i32;
    let pts = vec![
        Point::new(w / 2, h / 4),
        Point::new(3 * w / 4, h / 2),
        Point::new(w / 2, 3 * h / 4),
        Point::new(w / 4, h / 2),
    ];

    polylines(&mut result, &pts, true, color, 2)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: result })
}

/// Put text on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = putText)]
pub async fn put_text_wasm(src: &WasmMat, text: String, x: i32, y: i32, font_scale: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();
    let org = Point::new(x, y);
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);

    put_text(&mut result, &text, org, font_scale, color)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: result })
}
