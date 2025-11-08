use crate::dnn::blob::Blob;
use crate::dnn::layers::*;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Neural network for inference
pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    layer_map: HashMap<String, usize>,
    input_blob: Option<Blob>,
}

impl Network {
    /// Create new empty network
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            layer_map: HashMap::new(),
            input_blob: None,
        }
    }

    /// Add layer to network
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        let name = layer.name().to_string();
        let idx = self.layers.len();
        self.layers.push(layer);
        self.layer_map.insert(name, idx);
    }

    /// Set input blob
    pub fn set_input(&mut self, blob: Blob, name: Option<&str>) {
        self.input_blob = Some(blob);
    }

    /// Forward pass through the network
    pub fn forward(&self) -> Result<Blob> {
        if self.input_blob.is_none() {
            return Err(Error::InvalidParameter(
                "No input set. Call set_input first.".to_string()
            ));
        }

        let mut current = self.input_blob.as_ref().unwrap().clone_blob();

        for layer in &self.layers {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    /// Forward pass and return output from specific layer
    pub fn forward_to_layer(&self, layer_name: &str) -> Result<Blob> {
        if self.input_blob.is_none() {
            return Err(Error::InvalidParameter(
                "No input set. Call set_input first.".to_string()
            ));
        }

        let target_idx = self.layer_map.get(layer_name)
            .ok_or_else(|| Error::InvalidParameter(
                format!("Layer '{}' not found", layer_name)
            ))?;

        let mut current = self.input_blob.as_ref().unwrap().clone_blob();

        for (idx, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if idx == *target_idx {
                break;
            }
        }

        Ok(current)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer names
    pub fn get_layer_names(&self) -> Vec<String> {
        self.layers.iter().map(|l| l.name().to_string()).collect()
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// Load network from Caffe model
pub fn read_net_from_caffe(
    _proto_file: &str,
    _model_file: &str,
) -> Result<Network> {
    // In real implementation, would parse Caffe prototxt and binary model
    // For now, return empty network
    Ok(Network::new())
}

/// Load network from TensorFlow model
pub fn read_net_from_tensorflow(
    _model_file: &str,
) -> Result<Network> {
    // In real implementation, would parse TensorFlow model
    Ok(Network::new())
}

/// Load network from ONNX model
pub fn read_net_from_onnx(
    _model_file: &str,
) -> Result<Network> {
    // In real implementation, would parse ONNX model
    Ok(Network::new())
}

/// Load network from Torch model
pub fn read_net_from_torch(
    _model_file: &str,
) -> Result<Network> {
    // In real implementation, would parse Torch model
    Ok(Network::new())
}

/// Network builder for creating custom networks
pub struct NetworkBuilder {
    network: Network,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            network: Network::new(),
        }
    }

    pub fn add_conv(
        mut self,
        name: &str,
        num_filters: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let layer = ConvolutionLayer::new(
            name.to_string(),
            num_filters,
            kernel_size,
            stride,
            padding,
        );
        self.network.add_layer(Box::new(layer));
        self
    }

    pub fn add_pool(
        mut self,
        name: &str,
        pool_type: PoolType,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Self {
        let layer = PoolingLayer::new(
            name.to_string(),
            pool_type,
            kernel_size,
            stride,
        );
        self.network.add_layer(Box::new(layer));
        self
    }

    pub fn add_activation(
        mut self,
        name: &str,
        activation: ActivationType,
    ) -> Self {
        let layer = ActivationLayer::new(name.to_string(), activation);
        self.network.add_layer(Box::new(layer));
        self
    }

    pub fn add_fc(
        mut self,
        name: &str,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Self {
        let layer = FullyConnectedLayer::new(
            name.to_string(),
            num_inputs,
            num_outputs,
        );
        self.network.add_layer(Box::new(layer));
        self
    }

    pub fn add_flatten(mut self, name: &str) -> Self {
        let layer = FlattenLayer::new(name.to_string());
        self.network.add_layer(Box::new(layer));
        self
    }

    pub fn add_softmax(mut self, name: &str) -> Self {
        let layer = SoftmaxLayer::new(name.to_string());
        self.network.add_layer(Box::new(layer));
        self
    }

    pub fn build(self) -> Network {
        self.network
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Common network architectures
pub mod models {
    use super::*;

    /// Simple LeNet-5 like network
    pub fn lenet() -> Network {
        NetworkBuilder::new()
            .add_conv("conv1", 6, (5, 5), (1, 1), (0, 0))
            .add_activation("relu1", ActivationType::ReLU)
            .add_pool("pool1", PoolType::Max, (2, 2), (2, 2))
            .add_conv("conv2", 16, (5, 5), (1, 1), (0, 0))
            .add_activation("relu2", ActivationType::ReLU)
            .add_pool("pool2", PoolType::Max, (2, 2), (2, 2))
            .add_flatten("flatten")
            .add_fc("fc1", 400, 120)
            .add_activation("relu3", ActivationType::ReLU)
            .add_fc("fc2", 120, 84)
            .add_activation("relu4", ActivationType::ReLU)
            .add_fc("fc3", 84, 10)
            .add_softmax("softmax")
            .build()
    }

    /// Simple AlexNet-like network
    pub fn alexnet() -> Network {
        NetworkBuilder::new()
            .add_conv("conv1", 64, (11, 11), (4, 4), (2, 2))
            .add_activation("relu1", ActivationType::ReLU)
            .add_pool("pool1", PoolType::Max, (3, 3), (2, 2))
            .add_conv("conv2", 192, (5, 5), (1, 1), (2, 2))
            .add_activation("relu2", ActivationType::ReLU)
            .add_pool("pool2", PoolType::Max, (3, 3), (2, 2))
            .add_conv("conv3", 384, (3, 3), (1, 1), (1, 1))
            .add_activation("relu3", ActivationType::ReLU)
            .add_conv("conv4", 256, (3, 3), (1, 1), (1, 1))
            .add_activation("relu4", ActivationType::ReLU)
            .add_conv("conv5", 256, (3, 3), (1, 1), (1, 1))
            .add_activation("relu5", ActivationType::ReLU)
            .add_pool("pool3", PoolType::Max, (3, 3), (2, 2))
            .add_flatten("flatten")
            .add_fc("fc1", 9216, 4096)
            .add_activation("relu6", ActivationType::ReLU)
            .add_fc("fc2", 4096, 4096)
            .add_activation("relu7", ActivationType::ReLU)
            .add_fc("fc3", 4096, 1000)
            .add_softmax("softmax")
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = NetworkBuilder::new()
            .add_conv("conv1", 32, (3, 3), (1, 1), (1, 1))
            .add_activation("relu1", ActivationType::ReLU)
            .add_flatten("flatten")
            .build();

        assert_eq!(net.num_layers(), 3);
        let names = net.get_layer_names();
        assert_eq!(names[0], "conv1");
        assert_eq!(names[1], "relu1");
        assert_eq!(names[2], "flatten");
    }

    #[test]
    fn test_simple_forward() {
        let mut net = NetworkBuilder::new()
            .add_activation("relu", ActivationType::ReLU)
            .build();

        let input = Blob::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![1, 4]).unwrap();
        net.set_input(input, None);

        let output = net.forward().unwrap();
        assert_eq!(output.at(&[0, 0]).unwrap(), 0.0);
        assert_eq!(output.at(&[0, 2]).unwrap(), 1.0);
    }

    #[test]
    fn test_lenet_structure() {
        let net = models::lenet();
        assert!(net.num_layers() > 10);

        let names = net.get_layer_names();
        assert!(names.contains(&"conv1".to_string()));
        assert!(names.contains(&"softmax".to_string()));
    }
}
