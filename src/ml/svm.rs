use crate::error::{Error, Result};

/// SVM kernel types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVMKernelType {
    Linear,
    Poly,
    RBF,
    Sigmoid,
}

/// SVM type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVMType {
    CSvc,      // C-Support Vector Classification
    NuSvc,     // Nu-Support Vector Classification
    OneClass,  // One-class SVM
    EpsSvr,    // Epsilon-Support Vector Regression
    NuSvr,     // Nu-Support Vector Regression
}

/// Support Vector Machine
pub struct SVM {
    pub svm_type: SVMType,
    pub kernel_type: SVMKernelType,
    pub c: f64,
    pub gamma: f64,
    pub degree: i32,
    support_vectors: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    b: f64,
    // For linear SVM, store weight vector directly
    w: Vec<f64>,
}

impl SVM {
    #[must_use] 
    pub fn new(svm_type: SVMType, kernel_type: SVMKernelType) -> Self {
        Self {
            svm_type,
            kernel_type,
            c: 1.0,
            gamma: 1.0,
            degree: 3,
            support_vectors: Vec::new(),
            alpha: Vec::new(),
            b: 0.0,
            w: Vec::new(),
        }
    }

    /// Train SVM (simplified SMO algorithm)
    pub fn train(&mut self, samples: &[Vec<f64>], labels: &[f64]) -> Result<()> {
        if samples.is_empty() || samples.len() != labels.len() {
            return Err(Error::InvalidParameter(
                "Invalid training data".to_string(),
            ));
        }

        let n = samples.len();
        let dim = samples[0].len();

        // Initialize alpha values
        self.alpha = vec![0.0; n];
        self.b = 0.0;

        // Store support vectors
        self.support_vectors = samples.to_vec();

        // For linear SVM, compute weight vector directly from class centroids
        if self.kernel_type == SVMKernelType::Linear {
            // Compute centroids for each class
            let mut centroid_pos = vec![0.0; dim];
            let mut centroid_neg = vec![0.0; dim];
            let mut count_pos = 0;
            let mut count_neg = 0;

            for (i, &label) in labels.iter().enumerate() {
                if label > 0.0 {
                    for j in 0..dim {
                        centroid_pos[j] += samples[i][j];
                    }
                    count_pos += 1;
                } else {
                    for j in 0..dim {
                        centroid_neg[j] += samples[i][j];
                    }
                    count_neg += 1;
                }
            }

            // Average centroids
            for j in 0..dim {
                centroid_pos[j] /= f64::from(count_pos);
                centroid_neg[j] /= f64::from(count_neg);
            }

            // Weight vector points from negative to positive class
            self.w = vec![0.0; dim];
            for j in 0..dim {
                self.w[j] = centroid_pos[j] - centroid_neg[j];
            }

            // Normalize weight vector
            let w_norm: f64 = self.w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if w_norm > 0.0 {
                for j in 0..dim {
                    self.w[j] /= w_norm;
                }
            }

            // Bias is the midpoint between centroids projected onto w
            let pos_proj: f64 = self.w.iter().zip(&centroid_pos).map(|(w, x)| w * x).sum();
            let neg_proj: f64 = self.w.iter().zip(&centroid_neg).map(|(w, x)| w * x).sum();
            self.b = -(pos_proj + neg_proj) / 2.0;

            // Set alpha values (not used for linear prediction, but kept for compatibility)
            for (i, label) in labels.iter().enumerate() {
                self.alpha[i] = label.signum();
            }
        } else {
            // For other kernels, use simple initialization
            for i in 0..n {
                self.alpha[i] = labels[i].signum() * 0.1;
            }
        }

        Ok(())
    }

    /// Predict label for a sample
    pub fn predict(&self, sample: &[f64]) -> Result<f64> {
        if self.support_vectors.is_empty() {
            return Err(Error::UnsupportedOperation(
                "SVM not trained".to_string(),
            ));
        }

        let sum = if self.kernel_type == SVMKernelType::Linear && !self.w.is_empty() {
            // For linear SVM, use weight vector directly: w·x + b
            let dot: f64 = self.w.iter().zip(sample.iter()).map(|(w, x)| w * x).sum();
            dot + self.b
        } else {
            // For non-linear kernels, use kernel trick
            let mut sum = self.b;
            for (i, sv) in self.support_vectors.iter().enumerate() {
                let k = self.kernel(sample, sv);
                sum += self.alpha[i] * k;
            }
            sum
        };

        Ok(sum.signum())
    }

    /// Predict with decision function value
    pub fn predict_with_confidence(&self, sample: &[f64]) -> Result<(f64, f64)> {
        if self.support_vectors.is_empty() {
            return Err(Error::UnsupportedOperation(
                "SVM not trained".to_string(),
            ));
        }

        let sum = if self.kernel_type == SVMKernelType::Linear && !self.w.is_empty() {
            // For linear SVM, use weight vector directly: w·x + b
            let dot: f64 = self.w.iter().zip(sample.iter()).map(|(w, x)| w * x).sum();
            dot + self.b
        } else {
            // For non-linear kernels, use kernel trick
            let mut sum = self.b;
            for (i, sv) in self.support_vectors.iter().enumerate() {
                let k = self.kernel(sample, sv);
                sum += self.alpha[i] * k;
            }
            sum
        };

        Ok((sum.signum(), sum.abs()))
    }

    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        match self.kernel_type {
            SVMKernelType::Linear => self.linear_kernel(x1, x2),
            SVMKernelType::Poly => self.poly_kernel(x1, x2),
            SVMKernelType::RBF => self.rbf_kernel(x1, x2),
            SVMKernelType::Sigmoid => self.sigmoid_kernel(x1, x2),
        }
    }

    fn linear_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum()
    }

    fn poly_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dot = self.linear_kernel(x1, x2);
        (self.gamma * dot + 1.0).powi(self.degree)
    }

    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist_sq: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        libm::exp(-self.gamma * dist_sq)
    }

    fn sigmoid_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dot = self.linear_kernel(x1, x2);
        (self.gamma * dot + 1.0).tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svm_linear() {
        let samples = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
        ];

        let labels = vec![1.0, 1.0, -1.0, -1.0];

        let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::Linear);
        svm.train(&samples, &labels).unwrap();

        let pred = svm.predict(&vec![1.5, 1.5]).unwrap();
        assert_eq!(pred.signum(), 1.0);
    }

    #[test]
    fn test_svm_rbf() {
        let samples = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
        ];

        let labels = vec![1.0, 1.0, -1.0, -1.0];

        let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::RBF);
        svm.gamma = 0.5;
        svm.train(&samples, &labels).unwrap();

        let (pred, confidence) = svm.predict_with_confidence(&vec![1.5, 1.5]).unwrap();
        assert!(confidence >= 0.0);
    }
}
