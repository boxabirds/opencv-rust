//! Stitching operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Panorama stitcher (simplified demo)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = panoramaStitcher)]
pub async fn panorama_stitcher_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    // For single image, just add annotation
    let mut result = src.inner.clone();
    let text = "Panorama stitching demo".to_string();
    let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 255.0, 0.0, 255.0));

    Ok(WasmMat { inner: result })
}


/// Feather blender for stitching
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = featherBlender)]
pub async fn feather_blender_wasm(src: &WasmMat, blend_strength: f32) -> Result<WasmMat, JsValue> {
    // Simple alpha blending demo
    let mut result = src.inner.clone();
    
    // Apply feathering effect to edges
    for row in 0..result.rows() {
        for col in 0..result.cols() {
            let edge_dist = col.min(result.cols() - col).min(row).min(result.rows() - row) as f32;
            let alpha = (edge_dist * blend_strength).min(1.0);
            
            let pixel = result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            for ch in 0..result.channels() {
                pixel[ch] = (pixel[ch] as f32 * alpha) as u8;
            }
        }
    }

    Ok(WasmMat { inner: result })
}


#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = multibandBlender)]
pub async fn multiband_blender_wasm(src: &WasmMat, num_bands: usize) -> Result<WasmMat, JsValue> {
    use crate::stitching::blending::MultiBandBlender;
    use crate::core::{Mat, MatDepth};
    
    // Create two overlapping images for blending demo
    let w = src.inner.cols();
    let h = src.inner.rows();
    
    // Split image into left and right halves with overlap
    let overlap = w / 4;
    
    // Create left image (0 to w/2 + overlap)
    let mut left = Mat::new(h, w / 2 + overlap, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            if col < src.inner.cols() {
                let src_pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                let dst_pixel = left.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                for ch in 0..src.inner.channels() {
                    dst_pixel[ch] = src_pixel[ch];
                }
            }
        }
    }
    
    // Create right image (w/2 - overlap to w)
    let mut right = Mat::new(h, w / 2 + overlap, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            let src_col = (w / 2 - overlap) + col;
            if src_col < src.inner.cols() {
                let src_pixel = src.inner.at(row, src_col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                let dst_pixel = right.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                for ch in 0..src.inner.channels() {
                    dst_pixel[ch] = src_pixel[ch];
                }
            }
        }
    }
    
    // Create masks
    let mut mask_left = Mat::new(h, w / 2 + overlap, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut mask_right = Mat::new(h, w / 2 + overlap, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            let left_pix = mask_left.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            left_pix[0] = 255;
            
            let right_pix = mask_right.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            right_pix[0] = 255;
        }
    }
    
    // Apply multi-band blending
    let blender = MultiBandBlender::new(num_bands.max(1).min(6));
    let result = blender.blend(&[left, right], &[mask_left, mask_right])
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    Ok(WasmMat { inner: result })
}
