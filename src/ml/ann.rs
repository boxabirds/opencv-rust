use crate::error::{Error, Result};

/// Artificial Neural Network (Multi-Layer Perceptron)
pub struct AnnMlp {
    layer_sizes: Vec<usize>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    learning_rate: f64,
    activation: ActivationFunction,
    trained: bool,
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
}

impl AnnMlp {
    /// Create new neural network with specified layer sizes
    #[must_use] 
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        assert!(layer_sizes.len() >= 2, "Network must have at least input and output layers");

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Initialize weights and biases randomly
        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];

            let mut layer_weights = vec![vec![0.0; cols]; rows];
            let mut layer_biases = vec![0.0; rows];

            // Xavier initialization
            let scale = (2.0 / (cols + rows) as f64).sqrt();

            for row in 0..rows {
                for col in 0..cols {
                    layer_weights[row][col] = (rand_f64() - 0.5) * 2.0 * scale;
                }
                layer_biases[row] = (rand_f64() - 0.5) * 0.2;
            }

            weights.push(layer_weights);
            biases.push(layer_biases);
        }

        Self {
            layer_sizes,
            weights,
            biases,
            learning_rate: 0.01,
            activation: ActivationFunction::Sigmoid,
            trained: false,
        }
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate;
    }

    /// Set activation function
    pub fn set_activation_function(&mut self, activation: ActivationFunction) {
        self.activation = activation;
    }

    /// Train the network
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
        epochs: usize,
    ) -> Result<()> {
        if inputs.len() != outputs.len() {
            return Err(Error::InvalidParameter(
                "Input and output sizes must match".to_string(),
            ));
        }

        if inputs.is_empty() {
            return Err(Error::InvalidParameter("Empty training data".to_string()));
        }

        // Train using backpropagation
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in inputs.iter().zip(outputs.iter()) {
                if input.len() != self.layer_sizes[0] {
                    return Err(Error::InvalidParameter(format!(
                        "Input size mismatch: expected {}, got {}",
                        self.layer_sizes[0],
                        input.len()
                    )));
                }

                if target.len() != *self.layer_sizes.last().unwrap() {
                    return Err(Error::InvalidParameter(format!(
                        "Output size mismatch: expected {}, got {}",
                        self.layer_sizes.last().unwrap(),
                        target.len()
                    )));
                }

                // Forward pass
                let (activations, pre_activations) = self.forward_pass(input)?;

                // Calculate loss (MSE)
                let output = activations.last().unwrap();
                for (o, t) in output.iter().zip(target.iter()) {
                    let diff = o - t;
                    total_loss += diff * diff;
                }

                // Backward pass
                self.backward_pass(&activations, &pre_activations, target)?;
            }

            // Print progress
            if epoch % 100 == 0 {
                let avg_loss = total_loss / (inputs.len() * outputs[0].len()) as f64;
                // Could log: epoch, avg_loss
            }
        }

        self.trained = true;
        Ok(())
    }

    /// Predict output for given input
    pub fn predict(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.layer_sizes[0] {
            return Err(Error::InvalidParameter(format!(
                "Input size mismatch: expected {}, got {}",
                self.layer_sizes[0],
                input.len()
            )));
        }

        let (activations, _) = self.forward_pass(input)?;
        Ok(activations.last().unwrap().clone())
    }

    fn forward_pass(&self, input: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let mut activations = vec![input.to_vec()];
        let mut pre_activations = Vec::new();

        for layer in 0..self.weights.len() {
            let prev_activation = &activations[layer];
            let mut z = vec![0.0; self.weights[layer].len()];

            // Matrix multiplication: W * a + b
            for i in 0..self.weights[layer].len() {
                z[i] = self.biases[layer][i];
                for j in 0..self.weights[layer][i].len() {
                    z[i] += self.weights[layer][i][j] * prev_activation[j];
                }
            }

            pre_activations.push(z.clone());

            // Apply activation function
            let a: Vec<f64> = z.iter().map(|&x| self.activate(x)).collect();
            activations.push(a);
        }

        Ok((activations, pre_activations))
    }

    fn backward_pass(
        &mut self,
        activations: &[Vec<f64>],
        pre_activations: &[Vec<f64>],
        target: &[f64],
    ) -> Result<()> {
        let num_layers = self.weights.len();

        // Calculate output layer error
        let output = activations.last().unwrap();
        let mut delta: Vec<f64> = output
            .iter()
            .zip(target.iter())
            .zip(pre_activations.last().unwrap().iter())
            .map(|((&o, &t), &z)| (o - t) * self.activate_derivative(z))
            .collect();

        // Backpropagate error
        for layer in (0..num_layers).rev() {
            // Update weights and biases
            for i in 0..self.weights[layer].len() {
                for j in 0..self.weights[layer][i].len() {
                    let gradient = delta[i] * activations[layer][j];
                    self.weights[layer][i][j] -= self.learning_rate * gradient;
                }
                self.biases[layer][i] -= self.learning_rate * delta[i];
            }

            // Calculate delta for previous layer
            if layer > 0 {
                let mut new_delta = vec![0.0; self.weights[layer - 1].len()];

                for j in 0..self.weights[layer - 1].len() {
                    for i in 0..self.weights[layer].len() {
                        new_delta[j] += self.weights[layer][i][j] * delta[i];
                    }
                    new_delta[j] *= self.activate_derivative(pre_activations[layer - 1][j]);
                }

                delta = new_delta;
            }
        }

        Ok(())
    }

    fn activate(&self, x: f64) -> f64 {
        match self.activation {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
        }
    }

    fn activate_derivative(&self, x: f64) -> f64 {
        match self.activation {
            ActivationFunction::Sigmoid => {
                let s = self.activate(x);
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
        }
    }

    /// Save model weights
    #[must_use] 
    pub fn get_weights(&self) -> Vec<Vec<Vec<f64>>> {
        self.weights.clone()
    }

    /// Load model weights
    pub fn set_weights(&mut self, weights: Vec<Vec<Vec<f64>>>) {
        self.weights = weights;
        self.trained = true;
    }
}

// Simple pseudo-random number generator
static mut RAND_SEED: u64 = 12345;

fn rand_f64() -> f64 {
    unsafe {
        RAND_SEED = RAND_SEED.wrapping_mul(1_103_515_245).wrapping_add(12345);
        ((RAND_SEED / 65536) % 32768) as f64 / 32768.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ann_creation() {
        let ann = AnnMlp::new(vec![2, 4, 1]);
        assert_eq!(ann.layer_sizes, vec![2, 4, 1]);
        assert_eq!(ann.weights.len(), 2);
    }

    #[test]
    fn test_ann_forward_pass() {
        let ann = AnnMlp::new(vec![2, 3, 1]);
        let input = vec![0.5, 0.3];
        let result = ann.predict(&input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_ann_training() {
        let mut ann = AnnMlp::new(vec![2, 4, 1]);

        // XOR problem
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];

        let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let result = ann.train(&inputs, &outputs, 100);
        assert!(result.is_ok());
        assert!(ann.trained);
    }

    #[test]
    fn test_activation_functions() {
        let ann = AnnMlp::new(vec![2, 3, 1]);

        assert!((ann.activate(0.0) - 0.5).abs() < 0.01); // Sigmoid at 0
        assert!(ann.activate(10.0) > 0.9); // Sigmoid at large positive
        assert!(ann.activate(-10.0) < 0.1); // Sigmoid at large negative
    }

    #[test]
    fn test_set_learning_rate() {
        let mut ann = AnnMlp::new(vec![2, 3, 1]);
        ann.set_learning_rate(0.1);
        assert_eq!(ann.learning_rate, 0.1);
    }

    #[test]
    fn test_set_activation() {
        let mut ann = AnnMlp::new(vec![2, 3, 1]);
        ann.set_activation_function(ActivationFunction::ReLU);
        assert_eq!(ann.activation, ActivationFunction::ReLU);
    }
}
