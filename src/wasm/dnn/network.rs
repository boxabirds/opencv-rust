//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== loadNetwork =====
#[wasm_bindgen(js_name = loadNetwork)]
pub async fn load_network_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::dnn::network::Network;
    use crate::dnn::layers::{ConvolutionLayer, ActivationLayer, ActivationType};
    use crate::dnn::blob::Blob;
    
    // Create a simple demo network
    let mut network = Network::new();
    
    // Add a simple convolutional layer for demo
    // Note: This is a simplified demo, real networks would be loaded from files
    let conv_layer = ConvolutionLayer::new(
        "conv1".to_string(),
        16, // num_filters (output channels)
        (3, 3), // kernel size
        (1, 1), // stride
        (1, 1), // padding
    );
    network.add_layer(Box::new(conv_layer));
    
    // Add ReLU activation
    let relu = ActivationLayer::new(
        "relu1".to_string(),
        ActivationType::ReLU,
    );
    network.add_layer(Box::new(relu));
    
    // Convert image to blob
    let blob = Blob::from_image(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    // Visualize network architecture with text overlay
    use crate::imgproc::drawing::{rectangle, put_text};
    use crate::core::types::{Rect, Scalar, Point};
    
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);
    let bg_color = Scalar::new(0.0, 0.0, 0.0, 128.0);
    
    // Draw network architecture boxes
    let y_start = 50;
    let box_height = 40;
    let box_width = 150;
    
    // Input layer box
    let rect1 = Rect::new(50, y_start, box_width, box_height);
    let _ = rectangle(&mut result, rect1, bg_color, -1);
    let _ = rectangle(&mut result, rect1, color, 2);
    let _ = put_text(&mut result, "Input: 3ch", Point::new(60, y_start + 25), 0.5, color);
    
    // Conv layer box
    let rect2 = Rect::new(50, y_start + 60, box_width, box_height);
    let _ = rectangle(&mut result, rect2, bg_color, -1);
    let _ = rectangle(&mut result, rect2, color, 2);
    let _ = put_text(&mut result, "Conv: 16ch", Point::new(60, y_start + 85), 0.5, color);
    
    // ReLU layer box
    let rect3 = Rect::new(50, y_start + 120, box_width, box_height);
    let _ = rectangle(&mut result, rect3, bg_color, -1);
    let _ = rectangle(&mut result, rect3, color, 2);
    let _ = put_text(&mut result, "ReLU", Point::new(60, y_start + 145), 0.5, color);
    
    Ok(WasmMat { inner: result })
}


// ===== blobFromImage =====
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
        let text_y = (result.rows() - 10) as i32;
        let _ = put_text(&mut result, &ch_text, Point::new(x as i32 + 10, text_y), 0.5, Scalar::new(255.0, 255.0, 255.0, 255.0));
    }

    Ok(WasmMat { inner: result })
}


