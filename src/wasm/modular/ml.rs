//! Ml operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// SVM Classifier (demo with simple pattern detection)
/// TODO: Implementation needs to be fixed - currently stubbed
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = svmClassifier)]
pub async fn svm_classifier_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("SVM classifier not yet fully implemented"))
}


#[cfg(feature = "ml_experimental")]
/// Decision Tree Classifier
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = decisionTree)]
pub async fn decision_tree_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("decision_tree_wasm not yet fully implemented"))
}


#[cfg(feature = "ml_experimental")]
/// Random Forest Classifier
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = randomForest)]
pub async fn random_forest_wasm(src: &WasmMat, n_trees: usize) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("random_forest_wasm not yet fully implemented"))
}


#[cfg(feature = "ml_experimental")]
/// K-Nearest Neighbors
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = knn)]
pub async fn knn_wasm(src: &WasmMat, k: usize) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("knn_wasm not yet fully implemented"))
}


#[cfg(feature = "ml_experimental")]
/// Neural Network (MLP)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = neuralNetwork)]
pub async fn neural_network_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("neural_network_wasm not yet fully implemented"))
}


/// Cascade Classifier (face/object detection demo)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = cascadeClassifier)]
pub async fn cascade_classifier_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("cascade_classifier_wasm not yet fully implemented"))
}


#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = loadNetwork)]
pub async fn load_network_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("load_network_wasm not yet fully implemented"))
}


#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = blobFromImage)]
pub async fn blob_from_image_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::dnn::blob::Blob;
    use crate::imgproc::drawing::{rectangle, put_text};
    use crate::core::types::{Rect, Scalar, Point};
    
    // Convert image to blob (NCHW format)
    let blob = Blob::from_image(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let shape = blob.shape();
    
    // Visualize the blob transformation
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);
    let bg_color = Scalar::new(0.0, 0.0, 0.0, 180.0);
    
    // Draw info box
    let info_height = 120;
    let info_rect = Rect::new(10, 10, 250, info_height);
    let _ = rectangle(&mut result, info_rect, bg_color, -1);
    let _ = rectangle(&mut result, info_rect, color, 2);
    
    // Display blob info
    let _ = put_text(&mut result, "Blob Conversion", Point::new(20, 35), 0.6, color);
    
    let shape_text = format!("Shape: {:?}", shape);
    let _ = put_text(&mut result, &shape_text, Point::new(20, 60), 0.5, color);
    
    let format_text = "Format: NCHW";
    let _ = put_text(&mut result, format_text, Point::new(20, 85), 0.5, color);
    
    let norm_text = "Normalized: [0, 1]";
    let _ = put_text(&mut result, norm_text, Point::new(20, 110), 0.5, color);
    
    // Draw channel separation visualization
    let ch_width = result.cols() / 3;
    for i in 0..3 {
        let x = i * ch_width;
        let rect = Rect::new(x as i32, (result.rows() - 30) as i32, ch_width as i32, 25);
        let ch_color = match i {
            0 => Scalar::new(255.0, 0.0, 0.0, 255.0),
            1 => Scalar::new(0.0, 255.0, 0.0, 255.0),
            _ => Scalar::new(0.0, 0.0, 255.0, 255.0),
        };
        let _ = rectangle(&mut result, rect, ch_color, -1);
        
        let ch_text = format!("Ch{}", i);
        let _ = put_text(&mut result, &ch_text, Point::new(x as i32 + 10, (result.rows() - 10) as i32), 0.5, Scalar::new(255.0, 255.0, 255.0, 255.0));
    }

    Ok(WasmMat { inner: result })
}
