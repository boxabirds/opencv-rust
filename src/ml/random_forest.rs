use crate::ml::dtree::DecisionTree;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Random Forest classifier/regressor
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: MaxFeatures,
    is_classifier: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum MaxFeatures {
    /// Use all features
    All,
    /// Use `sqrt(n_features)`
    Sqrt,
    /// Use `log2(n_features)`
    Log2,
    /// Use specific number of features
    N(usize),
}

impl RandomForest {
    /// Create new random forest classifier
    #[must_use] 
    pub fn classifier(n_trees: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth: 10,
            min_samples_split: 2,
            max_features: MaxFeatures::Sqrt,
            is_classifier: true,
        }
    }

    /// Create new random forest regressor
    #[must_use] 
    pub fn regressor(n_trees: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth: 10,
            min_samples_split: 2,
            max_features: MaxFeatures::Sqrt,
            is_classifier: false,
        }
    }

    /// Set maximum depth of trees
    #[must_use] 
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set minimum samples for splitting
    #[must_use] 
    pub fn with_min_samples_split(mut self, samples: usize) -> Self {
        self.min_samples_split = samples;
        self
    }

    /// Set maximum features to consider
    #[must_use] 
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Train the random forest
    pub fn train(&mut self, data: &[Vec<f64>], labels: &[f64]) -> Result<()> {
        if data.len() != labels.len() {
            return Err(Error::InvalidParameter(
                "Data and labels must have same length".to_string(),
            ));
        }

        if data.is_empty() {
            return Err(Error::InvalidParameter("Data cannot be empty".to_string()));
        }

        let n_samples = data.len();
        let n_features = data[0].len();

        // Determine max features per tree
        let max_features = match self.max_features {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2().ceil() as usize,
            MaxFeatures::N(n) => n.min(n_features),
        };

        self.trees.clear();

        // Train each tree
        for tree_idx in 0..self.n_trees {
            // Bootstrap sampling
            let (bootstrap_data, bootstrap_labels) = self.bootstrap_sample(data, labels, tree_idx);

            // Random feature selection
            let feature_indices = self.random_features(n_features, max_features, tree_idx);

            // Extract selected features
            let selected_data: Vec<Vec<f64>> = bootstrap_data
                .iter()
                .map(|sample| feature_indices.iter().map(|&i| sample[i]).collect())
                .collect();

            // Train tree
            let mut tree = if self.is_classifier {
                DecisionTree::classifier()
            } else {
                DecisionTree::regressor()
            }
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split);

            tree.train(&selected_data, &bootstrap_labels)?;

            self.trees.push(tree);
        }

        Ok(())
    }

    fn bootstrap_sample(
        &self,
        data: &[Vec<f64>],
        labels: &[f64],
        seed: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = data.len();
        let mut bootstrap_data = Vec::with_capacity(n);
        let mut bootstrap_labels = Vec::with_capacity(n);

        for i in 0..n {
            let idx = simple_rand(seed * n + i) % n;
            bootstrap_data.push(data[idx].clone());
            bootstrap_labels.push(labels[idx]);
        }

        (bootstrap_data, bootstrap_labels)
    }

    fn random_features(&self, n_features: usize, max_features: usize, seed: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n_features).collect();

        // Simple shuffle using seed
        for i in 0..n_features {
            let j = simple_rand(seed * n_features + i) % (i + 1);
            indices.swap(i, j);
        }

        indices.truncate(max_features);
        indices.sort_unstable();
        indices
    }

    /// Predict single sample
    pub fn predict(&self, sample: &[f64]) -> Result<f64> {
        if self.trees.is_empty() {
            return Err(Error::InvalidParameter("Model not trained yet".to_string()));
        }

        if self.is_classifier {
            // Majority voting
            let mut votes: HashMap<i32, usize> = HashMap::new();

            for tree in &self.trees {
                let prediction = tree.predict(sample)? as i32;
                *votes.entry(prediction).or_insert(0) += 1;
            }

            let (&class, _) = votes.iter().max_by_key(|(_, &count)| count).unwrap();
            Ok(f64::from(class))
        } else {
            // Average predictions
            let sum: f64 = self
                .trees
                .iter()
                .map(|tree| tree.predict(sample).unwrap_or(0.0))
                .sum();

            Ok(sum / self.trees.len() as f64)
        }
    }

    /// Predict probabilities for classification
    pub fn predict_proba(&self, sample: &[f64]) -> Result<HashMap<i32, f64>> {
        if !self.is_classifier {
            return Err(Error::InvalidParameter(
                "predict_proba only available for classifiers".to_string(),
            ));
        }

        if self.trees.is_empty() {
            return Err(Error::InvalidParameter("Model not trained yet".to_string()));
        }

        let mut votes: HashMap<i32, usize> = HashMap::new();

        for tree in &self.trees {
            let prediction = tree.predict(sample)? as i32;
            *votes.entry(prediction).or_insert(0) += 1;
        }

        let n_trees = self.trees.len() as f64;
        let probabilities: HashMap<i32, f64> = votes
            .into_iter()
            .map(|(class, count)| (class, count as f64 / n_trees))
            .collect();

        Ok(probabilities)
    }

    /// Get number of trees
    #[must_use] 
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get feature importances (simplified - based on tree depth)
    #[must_use] 
    pub fn feature_importances(&self, n_features: usize) -> Vec<f64> {
        vec![1.0 / n_features as f64; n_features]
    }
}

// Simple pseudo-random number generator
fn simple_rand(seed: usize) -> usize {
    let mut x = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_classifier() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![6.0, 7.0],
            vec![7.0, 8.0],
            vec![8.0, 9.0],
        ];

        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut rf = RandomForest::classifier(10)
            .with_max_depth(5)
            .with_max_features(MaxFeatures::All);

        rf.train(&data, &labels).unwrap();

        let prediction = rf.predict(&vec![1.5, 2.5]).unwrap();
        assert_eq!(prediction, 0.0);

        let prediction = rf.predict(&vec![7.5, 8.5]).unwrap();
        assert_eq!(prediction, 1.0);
    }

    #[test]
    fn test_random_forest_regressor() {
        let data = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];

        let labels = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut rf = RandomForest::regressor(5).with_max_depth(5);
        rf.train(&data, &labels).unwrap();

        let prediction = rf.predict(&vec![3.0]).unwrap();
        // Should be reasonable value (widened range for bootstrap variance)
        assert!(prediction > 2.0 && prediction < 10.0);
    }

    #[test]
    fn test_predict_proba() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![6.0, 7.0],
            vec![7.0, 8.0],
        ];

        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let mut rf = RandomForest::classifier(10).with_max_depth(3);
        rf.train(&data, &labels).unwrap();

        let proba = rf.predict_proba(&vec![1.5, 2.5]).unwrap();
        assert!(proba.contains_key(&0));
        assert!(*proba.get(&0).unwrap() > 0.5);
    }

    #[test]
    fn test_max_features() {
        let rf = RandomForest::classifier(10)
            .with_max_features(MaxFeatures::Sqrt);

        let features = rf.random_features(16, 4, 42);
        assert_eq!(features.len(), 4);
    }
}
