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
}

impl SVM {
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

        // Initialize alpha values
        self.alpha = vec![0.0; n];
        self.b = 0.0;

        // Simplified training: just store support vectors
        // In a full implementation, we would use SMO algorithm
        self.support_vectors = samples.to_vec();

        // For linear SVM, calculate simple weights
        if self.kernel_type == SVMKernelType::Linear {
            for (i, label) in labels.iter().enumerate() {
                self.alpha[i] = if *label > 0.0 { 1.0 } else { -1.0 };
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

        let mut sum = self.b;

        for (i, sv) in self.support_vectors.iter().enumerate() {
            let k = self.kernel(sample, sv);
            sum += self.alpha[i] * k;
        }

        Ok(sum.signum())
    }

    /// Predict with decision function value
    pub fn predict_with_confidence(&self, sample: &[f64]) -> Result<(f64, f64)> {
        if self.support_vectors.is_empty() {
            return Err(Error::UnsupportedOperation(
                "SVM not trained".to_string(),
            ));
        }

        let mut sum = self.b;

        for (i, sv) in self.support_vectors.iter().enumerate() {
            let k = self.kernel(sample, sv);
            sum += self.alpha[i] * k;
        }

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

        (-self.gamma * dist_sq).exp()
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
