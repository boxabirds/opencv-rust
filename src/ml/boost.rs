#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::error::{Error, Result};

/// `AdaBoost` classifier
pub struct AdaBoostClassifier {
    weak_classifiers: Vec<WeakClassifier>,
    alphas: Vec<f64>,
    num_iterations: usize,
    trained: bool,
}

#[derive(Clone)]
struct WeakClassifier {
    feature_idx: usize,
    threshold: f64,
    polarity: i32, // 1 or -1
}

impl AdaBoostClassifier {
    /// Create new `AdaBoost` classifier
    #[must_use] 
    pub fn new(num_iterations: usize) -> Self {
        Self {
            weak_classifiers: Vec::new(),
            alphas: Vec::new(),
            num_iterations,
            trained: false,
        }
    }

    /// Train classifier
    pub fn train(&mut self, features: &[Vec<f64>], labels: &[i32]) -> Result<()> {
        if features.len() != labels.len() {
            return Err(Error::InvalidParameter(
                "Features and labels must have same length".to_string(),
            ));
        }

        if features.is_empty() {
            return Err(Error::InvalidParameter("Empty training data".to_string()));
        }

        let n_samples = features.len();
        let n_features = features[0].len();

        // Initialize weights uniformly
        let mut weights = vec![1.0 / n_samples as f64; n_samples];

        for _iteration in 0..self.num_iterations {
            // Find best weak classifier
            let (best_classifier, best_error) =
                self.find_best_weak_classifier(features, labels, &weights, n_features)?;

            if best_error >= 0.5 {
                break; // No improvement
            }

            // Calculate alpha
            let alpha = 0.5 * ((1.0 - best_error) / best_error.max(1e-10)).ln();

            // Update weights
            for i in 0..n_samples {
                let prediction = best_classifier.predict(&features[i]);
                let correct = if prediction == labels[i] { 1.0 } else { -1.0 };
                weights[i] *= libm::exp(-alpha * correct);
            }

            // Normalize weights
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }

            self.weak_classifiers.push(best_classifier);
            self.alphas.push(alpha);
        }

        self.trained = true;
        Ok(())
    }

    /// Predict class for input
    pub fn predict(&self, features: &[f64]) -> Result<i32> {
        if !self.trained {
            return Err(Error::InvalidParameter("Model not trained".to_string()));
        }

        let mut score = 0.0;

        for (classifier, &alpha) in self.weak_classifiers.iter().zip(self.alphas.iter()) {
            score += alpha * f64::from(classifier.predict(features));
        }

        Ok(if score >= 0.0 { 1 } else { -1 })
    }

    fn find_best_weak_classifier(
        &self,
        features: &[Vec<f64>],
        labels: &[i32],
        weights: &[f64],
        n_features: usize,
    ) -> Result<(WeakClassifier, f64)> {
        let mut best_classifier = WeakClassifier {
            feature_idx: 0,
            threshold: 0.0,
            polarity: 1,
        };
        let mut best_error = f64::INFINITY;

        // Try all features and thresholds
        for feature_idx in 0..n_features {
            // Get all values for this feature
            let mut values: Vec<f64> = features.iter().map(|f| f[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Try thresholds at midpoints
            for i in 0..values.len() - 1 {
                let threshold = f64::midpoint(values[i], values[i + 1]);

                // Try both polarities
                for &polarity in &[1, -1] {
                    let classifier = WeakClassifier {
                        feature_idx,
                        threshold,
                        polarity,
                    };

                    // Calculate weighted error
                    let mut error = 0.0;
                    for (i, feature) in features.iter().enumerate() {
                        let prediction = classifier.predict(feature);
                        if prediction != labels[i] {
                            error += weights[i];
                        }
                    }

                    if error < best_error {
                        best_error = error;
                        best_classifier = classifier;
                    }
                }
            }
        }

        Ok((best_classifier, best_error))
    }
}

impl WeakClassifier {
    fn predict(&self, features: &[f64]) -> i32 {
        let value = features[self.feature_idx];
        if (value < self.threshold) == (self.polarity == 1) {
            1
        } else {
            -1
        }
    }
}

/// Gradient Boosting regressor
pub struct GradientBoostingRegressor {
    trees: Vec<RegressionTree>,
    learning_rate: f64,
    num_iterations: usize,
    max_depth: usize,
    trained: bool,
    init_prediction: f64,
}

#[derive(Clone)]
struct RegressionTree {
    feature_idx: Option<usize>,
    threshold: Option<f64>,
    value: f64,
    left: Option<Box<RegressionTree>>,
    right: Option<Box<RegressionTree>>,
}

impl GradientBoostingRegressor {
    /// Create new gradient boosting regressor
    #[must_use] 
    pub fn new(num_iterations: usize, learning_rate: f64, max_depth: usize) -> Self {
        Self {
            trees: Vec::new(),
            learning_rate,
            num_iterations,
            max_depth,
            trained: false,
            init_prediction: 0.0,
        }
    }

    /// Train regressor
    pub fn train(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<()> {
        if features.len() != targets.len() {
            return Err(Error::InvalidParameter(
                "Features and targets must have same length".to_string(),
            ));
        }

        if features.is_empty() {
            return Err(Error::InvalidParameter("Empty training data".to_string()));
        }

        // Initialize with mean
        self.init_prediction = targets.iter().sum::<f64>() / targets.len() as f64;

        // Current predictions
        let mut predictions = vec![self.init_prediction; targets.len()];

        for _iteration in 0..self.num_iterations {
            // Calculate residuals (negative gradients)
            let residuals: Vec<f64> = targets
                .iter()
                .zip(predictions.iter())
                .map(|(&t, &p)| t - p)
                .collect();

            // Fit tree to residuals
            let tree = self.fit_tree(features, &residuals, 0)?;

            // Update predictions
            for (i, feature) in features.iter().enumerate() {
                predictions[i] += self.learning_rate * tree.predict(feature);
            }

            self.trees.push(tree);
        }

        self.trained = true;
        Ok(())
    }

    /// Predict target for input
    pub fn predict(&self, features: &[f64]) -> Result<f64> {
        if !self.trained {
            return Err(Error::InvalidParameter("Model not trained".to_string()));
        }

        let mut prediction = self.init_prediction;

        for tree in &self.trees {
            prediction += self.learning_rate * tree.predict(features);
        }

        Ok(prediction)
    }

    fn fit_tree(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        depth: usize,
    ) -> Result<RegressionTree> {
        // Base case: return leaf with mean value
        if depth >= self.max_depth || features.len() < 2 {
            let value = targets.iter().sum::<f64>() / targets.len() as f64;
            return Ok(RegressionTree {
                feature_idx: None,
                threshold: None,
                value,
                left: None,
                right: None,
            });
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) =
            self.find_best_split(features, targets)?;

        if best_gain <= 0.0 {
            let value = targets.iter().sum::<f64>() / targets.len() as f64;
            return Ok(RegressionTree {
                feature_idx: None,
                threshold: None,
                value,
                left: None,
                right: None,
            });
        }

        // Split data
        let (left_features, left_targets, right_features, right_targets) =
            self.split_data(features, targets, best_feature, best_threshold);

        // Recursively build subtrees
        let left_tree = self.fit_tree(&left_features, &left_targets, depth + 1)?;
        let right_tree = self.fit_tree(&right_features, &right_targets, depth + 1)?;

        Ok(RegressionTree {
            feature_idx: Some(best_feature),
            threshold: Some(best_threshold),
            value: 0.0,
            left: Some(Box::new(left_tree)),
            right: Some(Box::new(right_tree)),
        })
    }

    fn find_best_split(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<(usize, f64, f64)> {
        if features.is_empty() {
            return Ok((0, 0.0, 0.0));
        }

        let n_features = features[0].len();
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        let total_variance = calculate_variance(targets);

        for feature_idx in 0..n_features {
            let mut values: Vec<(f64, f64)> = features
                .iter()
                .zip(targets.iter())
                .map(|(f, &t)| (f[feature_idx], t))
                .collect();

            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for i in 0..values.len() - 1 {
                let threshold = f64::midpoint(values[i].0, values[i + 1].0);

                let (left_targets, right_targets): (Vec<f64>, Vec<f64>) =
                    values.iter().partition_map(|(v, t)| {
                        if *v <= threshold {
                            itertools::Either::Left(*t)
                        } else {
                            itertools::Either::Right(*t)
                        }
                    });

                if left_targets.is_empty() || right_targets.is_empty() {
                    continue;
                }

                let left_var = calculate_variance(&left_targets);
                let right_var = calculate_variance(&right_targets);

                let weighted_var = (left_targets.len() as f64 * left_var
                    + right_targets.len() as f64 * right_var)
                    / targets.len() as f64;

                let gain = total_variance - weighted_var;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        Ok((best_feature, best_threshold, best_gain))
    }

    fn split_data(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let mut left_features = Vec::new();
        let mut left_targets = Vec::new();
        let mut right_features = Vec::new();
        let mut right_targets = Vec::new();

        for (f, &t) in features.iter().zip(targets.iter()) {
            if f[feature_idx] <= threshold {
                left_features.push(f.clone());
                left_targets.push(t);
            } else {
                right_features.push(f.clone());
                right_targets.push(t);
            }
        }

        (left_features, left_targets, right_features, right_targets)
    }
}

impl RegressionTree {
    fn predict(&self, features: &[f64]) -> f64 {
        if let Some(feature_idx) = self.feature_idx {
            if features[feature_idx] <= self.threshold.unwrap() {
                self.left.as_ref().unwrap().predict(features)
            } else {
                self.right.as_ref().unwrap().predict(features)
            }
        } else {
            self.value
        }
    }
}

fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

// Helper trait for partition_map (simplified version)
trait PartitionMapExt {
    type Left;
    type Right;

    fn partition_map<F, L, R>(self, f: F) -> (Vec<L>, Vec<R>)
    where
        F: FnMut(&Self::Left) -> itertools::Either<L, R>;
}

impl<T> PartitionMapExt for std::slice::Iter<'_, T> {
    type Left = T;
    type Right = T;

    fn partition_map<F, L, R>(self, mut f: F) -> (Vec<L>, Vec<R>)
    where
        F: FnMut(&T) -> itertools::Either<L, R>,
    {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for item in self {
            match f(item) {
                itertools::Either::Left(l) => left.push(l),
                itertools::Either::Right(r) => right.push(r),
            }
        }

        (left, right)
    }
}

mod itertools {
    pub enum Either<L, R> {
        Left(L),
        Right(R),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaboost_creation() {
        let boost = AdaBoostClassifier::new(10);
        assert_eq!(boost.num_iterations, 10);
        assert!(!boost.trained);
    }

    #[test]
    fn test_adaboost_training() {
        let mut boost = AdaBoostClassifier::new(5);

        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ];

        let labels = vec![1, 1, -1, -1];

        let result = boost.train(&features, &labels);
        assert!(result.is_ok());
        assert!(boost.trained);
    }

    #[test]
    fn test_gradient_boosting_creation() {
        let gb = GradientBoostingRegressor::new(10, 0.1, 3);
        assert_eq!(gb.num_iterations, 10);
        assert_eq!(gb.learning_rate, 0.1);
        assert!(!gb.trained);
    }

    #[test]
    fn test_gradient_boosting_training() {
        let mut gb = GradientBoostingRegressor::new(5, 0.1, 2);

        let features = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];

        let targets = vec![2.0, 4.0, 6.0, 8.0];

        let result = gb.train(&features, &targets);
        assert!(result.is_ok());
        assert!(gb.trained);
    }

    #[test]
    fn test_gradient_boosting_predict() {
        let mut gb = GradientBoostingRegressor::new(10, 0.1, 3);

        let features = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![1.0, 2.0, 3.0];

        gb.train(&features, &targets).unwrap();

        let prediction = gb.predict(&vec![2.5]).unwrap();
        assert!(prediction > 0.0 && prediction < 10.0);
    }
}
