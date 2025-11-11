//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== svmClassifier =====
#[wasm_bindgen(js_name = svmClassifier)]
pub async fn svm_classifier_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::ml::svm::{SVM, SVMType, SVMKernelType};
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            // SVM classifier uses CPU implementation
        }
        cpu => {
            // Create simple training data (bright vs dark regions)
            let mut train_data = Vec::new();
            let mut labels = Vec::new();

            // Sample from image
            for row in (0..src.inner.rows()).step_by(20) {
                for col in (0..src.inner.cols()).step_by(20) {
                    let pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                    let intensity = pixel[0] as f64;
                    train_data.push(vec![intensity]);
                    labels.push(if intensity > 128.0 { 1.0 } else { -1.0 });
                }
            }

            // Train SVM
            let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::RBF);
            svm.train(&train_data, &labels)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Visualize classification
            let text = format!("SVM: {} samples", train_data.len());
            let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(0.0, 255.0, 0.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== decisionTree =====
#[wasm_bindgen(js_name = decisionTree)]
pub async fn decision_tree_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::ml::dtree::DecisionTree;
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            // Decision tree uses CPU implementation
        }
        cpu => {
            // Create simple training data
            let mut train_data = Vec::new();
            let mut labels = Vec::new();

            for row in (0..src.inner.rows()).step_by(20) {
                for col in (0..src.inner.cols()).step_by(20) {
                    let pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                    let intensity = pixel[0] as f64;
                    train_data.push(vec![intensity]);
                    labels.push(if intensity > 128.0 { 1.0 } else { 0.0 });
                }
            }

            // Train decision tree
            let mut tree = DecisionTree::classifier();
            tree.train(&train_data, &labels)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Visualize
            let text = format!("DTree: {} samples", train_data.len());
            let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 0.0, 0.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== randomForest =====
#[wasm_bindgen(js_name = randomForest)]
pub async fn random_forest_wasm(src: &WasmMat, n_trees: usize) -> Result<WasmMat, JsValue> {
    use crate::ml::random_forest::RandomForest;
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            // Random forest uses CPU implementation
        }
        cpu => {
            // Create training data
            let mut train_data = Vec::new();
            let mut labels = Vec::new();

            for row in (0..src.inner.rows()).step_by(20) {
                for col in (0..src.inner.cols()).step_by(20) {
                    let pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                    let intensity = pixel[0] as f64;
                    train_data.push(vec![intensity]);
                    labels.push(if intensity > 128.0 { 1.0 } else { 0.0 });
                }
            }

            // Train random forest
            let mut rf = RandomForest::classifier(n_trees);
            rf.train(&train_data, &labels)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Visualize
            let text = format!("RF: {} trees", n_trees);
            let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(0.0, 255.0, 255.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== knn =====
#[wasm_bindgen(js_name = knn)]
pub async fn knn_wasm(src: &WasmMat, k: usize) -> Result<WasmMat, JsValue> {
    use crate::ml::knearest::KNearest;
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            // KNN uses CPU implementation
        }
        cpu => {
            // Create training data
            let mut train_data = Vec::new();
            let mut labels = Vec::new();

            for row in (0..src.inner.rows()).step_by(20) {
                for col in (0..src.inner.cols()).step_by(20) {
                    let pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                    let intensity = pixel[0] as f64;
                    train_data.push(vec![intensity]);
                    labels.push(if intensity > 128.0 { 1.0 } else { 0.0 });
                }
            }

            // Train KNN
            let mut knn_model = KNearest::classifier(k);
            knn_model.train(&train_data, &labels)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Visualize
            let text = format!("KNN: k={}", k);
            let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 255.0, 0.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== neuralNetwork =====
#[wasm_bindgen(js_name = neuralNetwork)]
pub async fn neural_network_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::ml::ann::AnnMlp;
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            // Neural network uses CPU implementation
        }
        cpu => {
            // Create simple training data
            let mut train_data = Vec::new();
            let mut labels = Vec::new();

            for row in (0..src.inner.rows()).step_by(20) {
                for col in (0..src.inner.cols()).step_by(20) {
                    let pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                    let intensity = pixel[0] as f64 / 255.0;
                    train_data.push(vec![intensity]);
                    labels.push(vec![if intensity > 0.5 { 1.0 } else { 0.0 }]);
                }
            }

            // Train neural network
            let layer_sizes = vec![1, 5, 1];
            let mut nn = AnnMlp::new(layer_sizes);
            nn.train(&train_data, &labels, 100)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Visualize
            let text = "MLP: 1-5-1".to_string();
            let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 128.0, 0.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}


