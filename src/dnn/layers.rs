use crate::dnn::blob::Blob;
use crate::error::{Error, Result};

/// Base trait for neural network layers
pub trait Layer {
    /// Forward pass
    fn forward(&self, input: &Blob) -> Result<Blob>;

    /// Get layer name
    fn name(&self) -> &str;

    /// Get layer type
    fn layer_type(&self) -> LayerType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Convolution,
    Pooling,
    FullyConnected,
    Activation,
    BatchNorm,
    Dropout,
    Flatten,
    Softmax,
}

/// Convolution layer
pub struct ConvolutionLayer {
    name: String,
    weights: Blob,
    bias: Option<Blob>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    num_filters: usize,
}

impl ConvolutionLayer {
    pub fn new(
        name: String,
        num_filters: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        // Initialize weights (simplified - would use proper initialization)
        let weights = Blob::new(vec![num_filters, 1, kernel_size.0, kernel_size.1]);

        Self {
            name,
            weights,
            bias: None,
            kernel_size,
            stride,
            padding,
            num_filters,
        }
    }

    pub fn with_weights(mut self, weights: Blob, bias: Option<Blob>) -> Self {
        self.weights = weights;
        self.bias = bias;
        self
    }
}

impl Layer for ConvolutionLayer {
    fn forward(&self, input: &Blob) -> Result<Blob> {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(Error::InvalidDimensions(
                "Input must be 4D (NCHW)".to_string()
            ));
        }

        let batch = input_shape[0];
        let _in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        // Calculate output dimensions
        let out_height = (in_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_width = (in_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = Blob::new(vec![batch, self.num_filters, out_height, out_width]);

        // Simplified convolution (not optimized)
        for b in 0..batch {
            for f in 0..self.num_filters {
                for out_y in 0..out_height {
                    for out_x in 0..out_width {
                        let mut sum = 0.0;

                        // Convolve
                        for ky in 0..self.kernel_size.0 {
                            for kx in 0..self.kernel_size.1 {
                                let in_y = out_y * self.stride.0 + ky;
                                let in_x = out_x * self.stride.1 + kx;

                                if in_y < self.padding.0 || in_x < self.padding.1 {
                                    continue;
                                }

                                let in_y = in_y - self.padding.0;
                                let in_x = in_x - self.padding.1;

                                if in_y >= in_height || in_x >= in_width {
                                    continue;
                                }

                                let input_val = input.at(&[b, 0, in_y, in_x])?;
                                let weight_val = self.weights.at(&[f, 0, ky, kx])?;
                                sum += input_val * weight_val;
                            }
                        }

                        // Add bias if present
                        if let Some(ref bias) = self.bias {
                            sum += bias.at(&[f])?;
                        }

                        output.set(&[b, f, out_y, out_x], sum)?;
                    }
                }
            }
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Convolution
    }
}

/// Pooling layer
pub struct PoolingLayer {
    name: String,
    pool_type: PoolType,
    kernel_size: (usize, usize),
    stride: (usize, usize),
}

#[derive(Debug, Clone, Copy)]
pub enum PoolType {
    Max,
    Average,
}

impl PoolingLayer {
    pub fn new(
        name: String,
        pool_type: PoolType,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Self {
        Self {
            name,
            pool_type,
            kernel_size,
            stride,
        }
    }
}

impl Layer for PoolingLayer {
    fn forward(&self, input: &Blob) -> Result<Blob> {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(Error::InvalidDimensions(
                "Input must be 4D (NCHW)".to_string()
            ));
        }

        let batch = input_shape[0];
        let channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let out_height = (in_height - self.kernel_size.0) / self.stride.0 + 1;
        let out_width = (in_width - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = Blob::new(vec![batch, channels, out_height, out_width]);

        for b in 0..batch {
            for c in 0..channels {
                for out_y in 0..out_height {
                    for out_x in 0..out_width {
                        let start_y = out_y * self.stride.0;
                        let start_x = out_x * self.stride.1;

                        let value = match self.pool_type {
                            PoolType::Max => {
                                let mut max_val = f32::NEG_INFINITY;
                                for ky in 0..self.kernel_size.0 {
                                    for kx in 0..self.kernel_size.1 {
                                        let val = input.at(&[b, c, start_y + ky, start_x + kx])?;
                                        max_val = max_val.max(val);
                                    }
                                }
                                max_val
                            }
                            PoolType::Average => {
                                let mut sum = 0.0;
                                for ky in 0..self.kernel_size.0 {
                                    for kx in 0..self.kernel_size.1 {
                                        sum += input.at(&[b, c, start_y + ky, start_x + kx])?;
                                    }
                                }
                                sum / (self.kernel_size.0 * self.kernel_size.1) as f32
                            }
                        };

                        output.set(&[b, c, out_y, out_x], value)?;
                    }
                }
            }
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Pooling
    }
}

/// Activation layer
pub struct ActivationLayer {
    name: String,
    activation: ActivationType,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f32),
}

impl ActivationLayer {
    pub fn new(name: String, activation: ActivationType) -> Self {
        Self { name, activation }
    }
}

impl Layer for ActivationLayer {
    fn forward(&self, input: &Blob) -> Result<Blob> {
        let mut output = input.clone_blob();
        let data = output.data_mut();

        match self.activation {
            ActivationType::ReLU => {
                for val in data.iter_mut() {
                    *val = val.max(0.0);
                }
            }
            ActivationType::Sigmoid => {
                for val in data.iter_mut() {
                    *val = 1.0 / (1.0 + (-*val).exp());
                }
            }
            ActivationType::Tanh => {
                for val in data.iter_mut() {
                    *val = val.tanh();
                }
            }
            ActivationType::LeakyReLU(alpha) => {
                for val in data.iter_mut() {
                    *val = if *val > 0.0 { *val } else { alpha * *val };
                }
            }
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Activation
    }
}

/// Fully connected (dense) layer
pub struct FullyConnectedLayer {
    name: String,
    weights: Blob,
    bias: Option<Blob>,
    num_outputs: usize,
}

impl FullyConnectedLayer {
    pub fn new(name: String, num_inputs: usize, num_outputs: usize) -> Self {
        let weights = Blob::new(vec![num_outputs, num_inputs]);

        Self {
            name,
            weights,
            bias: None,
            num_outputs,
        }
    }

    pub fn with_weights(mut self, weights: Blob, bias: Option<Blob>) -> Self {
        self.weights = weights;
        self.bias = bias;
        self
    }
}

impl Layer for FullyConnectedLayer {
    fn forward(&self, input: &Blob) -> Result<Blob> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let num_inputs: usize = input_shape[1..].iter().product();

        let mut output = Blob::new(vec![batch_size, self.num_outputs]);

        for b in 0..batch_size {
            for out_idx in 0..self.num_outputs {
                let mut sum = 0.0;

                for in_idx in 0..num_inputs {
                    let input_val = input.data()[b * num_inputs + in_idx];
                    let weight_val = self.weights.at(&[out_idx, in_idx])?;
                    sum += input_val * weight_val;
                }

                if let Some(ref bias) = self.bias {
                    sum += bias.at(&[out_idx])?;
                }

                output.set(&[b, out_idx], sum)?;
            }
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn layer_type(&self) -> LayerType {
        LayerType::FullyConnected
    }
}

/// Flatten layer
pub struct FlattenLayer {
    name: String,
}

impl FlattenLayer {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl Layer for FlattenLayer {
    fn forward(&self, input: &Blob) -> Result<Blob> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let flattened_size: usize = input_shape[1..].iter().product();

        let mut output = input.clone_blob();
        output.reshape(vec![batch_size, flattened_size])?;

        Ok(output)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Flatten
    }
}

/// Softmax layer
pub struct SoftmaxLayer {
    name: String,
}

impl SoftmaxLayer {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl Layer for SoftmaxLayer {
    fn forward(&self, input: &Blob) -> Result<Blob> {
        let mut output = input.clone_blob();
        let shape = output.shape().to_vec();

        if shape.len() != 2 {
            return Err(Error::InvalidDimensions(
                "Softmax expects 2D input (batch, features)".to_string()
            ));
        }

        let batch_size = shape[0];
        let num_classes = shape[1];

        for b in 0..batch_size {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..num_classes {
                let val = output.at(&[b, i])?;
                max_val = max_val.max(val);
            }

            // Compute exp and sum
            let mut sum = 0.0;
            for i in 0..num_classes {
                let val = output.at(&[b, i])?;
                let exp_val = (val - max_val).exp();
                output.set(&[b, i], exp_val)?;
                sum += exp_val;
            }

            // Normalize
            for i in 0..num_classes {
                let val = output.at(&[b, i])?;
                output.set(&[b, i], val / sum)?;
            }
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Softmax
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_relu() {
        let layer = ActivationLayer::new("relu1".to_string(), ActivationType::ReLU);

        let input = Blob::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![1, 4]).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.at(&[0, 0]).unwrap(), 0.0);
        assert_eq!(output.at(&[0, 1]).unwrap(), 0.0);
        assert_eq!(output.at(&[0, 2]).unwrap(), 1.0);
        assert_eq!(output.at(&[0, 3]).unwrap(), 2.0);
    }

    #[test]
    fn test_flatten() {
        let layer = FlattenLayer::new("flatten".to_string());

        let input = Blob::new(vec![2, 3, 4, 5]);
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 60]); // 3*4*5 = 60
    }

    #[test]
    fn test_softmax() {
        let layer = SoftmaxLayer::new("softmax".to_string());

        let input = Blob::from_data(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let output = layer.forward(&input).unwrap();

        // Sum should be 1.0
        let sum = output.at(&[0, 0]).unwrap()
            + output.at(&[0, 1]).unwrap()
            + output.at(&[0, 2]).unwrap();

        assert!((sum - 1.0).abs() < 1e-6);
    }
}
