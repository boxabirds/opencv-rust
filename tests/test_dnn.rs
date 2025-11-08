// DNN tests ported from OpenCV test suite
// opencv/modules/dnn/test/test_layers.cpp
// opencv/modules/dnn/test/test_caffe_importer.cpp
// opencv/modules/dnn/test/test_tf_importer.cpp

use opencv_rust::dnn::blob::Blob;
use opencv_rust::dnn::layers::*;
use opencv_rust::dnn::network::*;

/// Test from opencv test_layers.cpp - blob creation
#[test]
fn test_blob_creation() {
    let blob = Blob::new(vec![1, 3, 224, 224]);

    assert_eq!(blob.shape(), &[1, 3, 224, 224]);
    assert_eq!(blob.total(), 1 * 3 * 224 * 224);
}

/// Test blob cloning from opencv test_layers.cpp
#[test]
fn test_blob_clone() {
    let mut blob1 = Blob::new(vec![2, 3, 4, 5]);
    blob1.set(&[0, 0, 0, 0], 42.0).unwrap();

    let blob2 = blob1.clone_blob();

    assert_eq!(blob2.shape(), blob1.shape());
    assert_eq!(blob2.at(&[0, 0, 0, 0]).unwrap(), 42.0);
}

/// Test blob data access from opencv test_layers.cpp
#[test]
fn test_blob_data_access() {
    let mut blob = Blob::new(vec![2, 3]);

    // Fill using data_mut
    for val in blob.data_mut() {
        *val = 3.14;
    }

    for i in 0..2 {
        for j in 0..3 {
            assert!((blob.at(&[i, j]).unwrap() - 3.14).abs() < 1e-6);
        }
    }
}

/// Test blob access with invalid indices from opencv test_layers.cpp
#[test]
fn test_blob_invalid_access() {
    let blob = Blob::new(vec![2, 3, 4]);

    // Out of bounds access should error
    let result = blob.at(&[2, 3, 4]);
    assert!(result.is_err());

    let result = blob.at(&[0, 0]);
    assert!(result.is_err(), "Should reject wrong number of dimensions");
}

/// Test from opencv test_layers.cpp - convolution layer creation
#[test]
fn test_convolution_layer_creation() {
    let layer = ConvolutionLayer::new(
        "conv1".to_string(),
        32,                // num_filters
        (3, 3),           // kernel_size
        (1, 1),           // stride
        (1, 1),           // padding
    );

    assert_eq!(layer.name(), "conv1");
    assert_eq!(layer.layer_type(), LayerType::Convolution);
}

/// Test convolution forward pass from opencv test_layers.cpp
#[test]
fn test_convolution_forward() {
    let layer = ConvolutionLayer::new(
        "conv".to_string(),
        1,                 // 1 filter
        (3, 3),
        (1, 1),
        (0, 0),           // no padding
    );

    // Input: 1 batch, 1 channel, 5x5 image
    let mut input = Blob::new(vec![1, 1, 5, 5]);
    for val in input.data_mut() {
        *val = 1.0;
    }

    let output = layer.forward(&input).unwrap();

    // Output should be 1x1x3x3 (no padding, stride 1, kernel 3x3)
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
}

/// Test convolution with padding from opencv test_layers.cpp
#[test]
fn test_convolution_with_padding() {
    let layer = ConvolutionLayer::new(
        "conv".to_string(),
        1,
        (3, 3),
        (1, 1),
        (1, 1),           // padding
    );

    let input = Blob::new(vec![1, 1, 5, 5]);

    let output = layer.forward(&input).unwrap();

    // With padding=1, output should be same size as input
    assert_eq!(output.shape(), &[1, 1, 5, 5]);
}

/// Test pooling layer from opencv test_layers.cpp
#[test]
fn test_pooling_layer_max() {
    let layer = PoolingLayer::new(
        "pool1".to_string(),
        PoolType::Max,
        (2, 2),
        (2, 2),
    );

    assert_eq!(layer.name(), "pool1");
    assert_eq!(layer.layer_type(), LayerType::Pooling);

    // Input: 1 batch, 1 channel, 4x4 image
    let mut input = Blob::new(vec![1, 1, 4, 4]);

    // Fill with pattern where max values are easy to identify
    for i in 0..4 {
        for j in 0..4 {
            input.set(&[0, 0, i, j], ((i * 4 + j) as f32)).unwrap();
        }
    }

    let output = layer.forward(&input).unwrap();

    // Max pooling 2x2 -> output should be 2x2
    assert_eq!(output.shape(), &[1, 1, 2, 2]);

    // Check max values were selected
    // Top-left 2x2 pool should select max of [0,1,4,5] = 5
    assert_eq!(output.at(&[0, 0, 0, 0]).unwrap(), 5.0);
}

/// Test average pooling from opencv test_layers.cpp
#[test]
fn test_pooling_layer_average() {
    let layer = PoolingLayer::new(
        "pool_avg".to_string(),
        PoolType::Average,
        (2, 2),
        (2, 2),
    );

    let mut input = Blob::new(vec![1, 1, 4, 4]);
    for val in input.data_mut() {
        *val = 4.0;
    }

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2, 2]);

    // Average of all 4.0 values should be 4.0
    assert_eq!(output.at(&[0, 0, 0, 0]).unwrap(), 4.0);
}

/// Test activation layer ReLU from opencv test_layers.cpp
#[test]
fn test_activation_relu() {
    let layer = ActivationLayer::new("relu".to_string(), ActivationType::ReLU);

    let mut input = Blob::new(vec![1, 1, 2, 2]);
    input.set(&[0, 0, 0, 0], 1.0).unwrap();
    input.set(&[0, 0, 0, 1], -1.0).unwrap();
    input.set(&[0, 0, 1, 0], 0.5).unwrap();
    input.set(&[0, 0, 1, 1], -0.5).unwrap();

    let output = layer.forward(&input).unwrap();

    // ReLU: max(0, x)
    assert_eq!(output.at(&[0, 0, 0, 0]).unwrap(), 1.0);
    assert_eq!(output.at(&[0, 0, 0, 1]).unwrap(), 0.0);  // -1 -> 0
    assert_eq!(output.at(&[0, 0, 1, 0]).unwrap(), 0.5);
    assert_eq!(output.at(&[0, 0, 1, 1]).unwrap(), 0.0);  // -0.5 -> 0
}

/// Test activation layer Sigmoid from opencv test_layers.cpp
#[test]
fn test_activation_sigmoid() {
    let layer = ActivationLayer::new("sigmoid".to_string(), ActivationType::Sigmoid);

    let mut input = Blob::new(vec![1, 1, 1, 1]);
    input.set(&[0, 0, 0, 0], 0.0).unwrap();

    let output = layer.forward(&input).unwrap();

    // Sigmoid(0) = 0.5
    assert!((output.at(&[0, 0, 0, 0]).unwrap() - 0.5).abs() < 0.01);
}

/// Test fully connected layer from opencv test_layers.cpp
#[test]
fn test_fully_connected_layer() {
    let layer = FullyConnectedLayer::new(
        "fc1".to_string(),
        4,  // input features
        2,  // output features
    );

    assert_eq!(layer.name(), "fc1");

    // Input: 1 batch, 4 features
    let input = Blob::new(vec![1, 4]);

    let output = layer.forward(&input).unwrap();

    // Output should have 2 features
    assert_eq!(output.shape(), &[1, 2]);
}

/// Test flatten layer from opencv test_layers.cpp
#[test]
fn test_flatten_layer() {
    let layer = FlattenLayer::new("flatten".to_string());

    // Input: 2 batch, 3 channels, 4 height, 5 width
    let input = Blob::new(vec![2, 3, 4, 5]);

    let output = layer.forward(&input).unwrap();

    // Output should be 2 x (3*4*5) = 2 x 60
    assert_eq!(output.shape(), &[2, 60]);
}

/// Test softmax layer from opencv test_layers.cpp
#[test]
fn test_softmax_layer() {
    let layer = SoftmaxLayer::new("softmax".to_string());

    let mut input = Blob::new(vec![1, 3]);
    input.set(&[0, 0], 1.0).unwrap();
    input.set(&[0, 1], 2.0).unwrap();
    input.set(&[0, 2], 3.0).unwrap();

    let output = layer.forward(&input).unwrap();

    // Softmax output should sum to 1
    let sum = output.at(&[0, 0]).unwrap()
        + output.at(&[0, 1]).unwrap()
        + output.at(&[0, 2]).unwrap();

    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1");

    // Largest input should have largest output
    assert!(output.at(&[0, 2]).unwrap() > output.at(&[0, 1]).unwrap());
    assert!(output.at(&[0, 1]).unwrap() > output.at(&[0, 0]).unwrap());
}

/// Test network creation from opencv test_caffe_importer.cpp
#[test]
fn test_network_creation() {
    let net = Network::new();

    assert_eq!(net.num_layers(), 0);
    assert_eq!(net.get_layer_names().len(), 0);
}

/// Test adding layers to network from opencv test_caffe_importer.cpp
#[test]
fn test_network_add_layers() {
    let mut net = Network::new();

    let layer1 = Box::new(ActivationLayer::new("relu1".to_string(), ActivationType::ReLU));
    let layer2 = Box::new(ActivationLayer::new("relu2".to_string(), ActivationType::ReLU));

    net.add_layer(layer1);
    net.add_layer(layer2);

    assert_eq!(net.num_layers(), 2);

    let names = net.get_layer_names();
    assert_eq!(names.len(), 2);
    assert_eq!(names[0], "relu1");
    assert_eq!(names[1], "relu2");
}

/// Test network forward without input from opencv test_caffe_importer.cpp
#[test]
fn test_network_forward_no_input() {
    let net = Network::new();

    let result = net.forward();

    assert!(result.is_err(), "Forward should fail without input");
}

/// Test network forward with input from opencv test_caffe_importer.cpp
#[test]
fn test_network_forward_with_input() {
    let mut net = Network::new();

    let relu = Box::new(ActivationLayer::new("relu".to_string(), ActivationType::ReLU));
    net.add_layer(relu);

    let mut input = Blob::new(vec![1, 1, 2, 2]);
    for val in input.data_mut() {
        *val = -1.0;
    }

    net.set_input(input, None);

    let output = net.forward().unwrap();

    // ReLU should convert all -1.0 to 0.0
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(output.at(&[0, 0, i, j]).unwrap(), 0.0);
        }
    }
}

/// Test network forward to specific layer from opencv test_caffe_importer.cpp
#[test]
fn test_network_forward_to_layer() {
    let mut net = Network::new();

    let layer1 = Box::new(ActivationLayer::new("layer1".to_string(), ActivationType::ReLU));
    let layer2 = Box::new(ActivationLayer::new("layer2".to_string(), ActivationType::ReLU));

    net.add_layer(layer1);
    net.add_layer(layer2);

    let input = Blob::new(vec![1, 1, 2, 2]);
    net.set_input(input, None);

    let output = net.forward_to_layer("layer1").unwrap();

    // Should have output from first layer only
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
}

/// Test network forward to non-existent layer from opencv test_caffe_importer.cpp
#[test]
fn test_network_forward_to_invalid_layer() {
    let mut net = Network::new();

    let layer = Box::new(ActivationLayer::new("layer1".to_string(), ActivationType::ReLU));
    net.add_layer(layer);

    let input = Blob::new(vec![1, 1, 2, 2]);
    net.set_input(input, None);

    let result = net.forward_to_layer("nonexistent");

    assert!(result.is_err(), "Should fail for non-existent layer");
}
