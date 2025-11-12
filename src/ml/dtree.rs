use crate::error::{Error, Result};
use std::collections::HashMap;

/// Decision tree for classification and regression
pub struct DecisionTree {
    root: Option<Box<TreeNode>>,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    is_classifier: bool,
}

struct TreeNode {
    feature_index: Option<usize>,
    threshold: Option<f64>,
    value: Option<f64>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    samples: usize,
}

impl DecisionTree {
    /// Create new decision tree classifier
    #[must_use] 
    pub fn classifier() -> Self {
        Self {
            root: None,
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            is_classifier: true,
        }
    }

    /// Create new decision tree regressor
    #[must_use] 
    pub fn regressor() -> Self {
        Self {
            root: None,
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            is_classifier: false,
        }
    }

    /// Set maximum depth of the tree
    #[must_use] 
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set minimum samples required to split a node
    #[must_use] 
    pub fn with_min_samples_split(mut self, samples: usize) -> Self {
        self.min_samples_split = samples;
        self
    }

    /// Set minimum samples required in a leaf node
    #[must_use] 
    pub fn with_min_samples_leaf(mut self, samples: usize) -> Self {
        self.min_samples_leaf = samples;
        self
    }

    /// Train the decision tree
    pub fn train(&mut self, data: &[Vec<f64>], labels: &[f64]) -> Result<()> {
        if data.len() != labels.len() {
            return Err(Error::InvalidParameter(
                "Data and labels must have same length".to_string(),
            ));
        }

        if data.is_empty() {
            return Err(Error::InvalidParameter("Data cannot be empty".to_string()));
        }

        let indices: Vec<usize> = (0..data.len()).collect();
        self.root = Some(self.build_tree(data, labels, &indices, 0)?);

        Ok(())
    }

    fn build_tree(
        &self,
        data: &[Vec<f64>],
        labels: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> Result<Box<TreeNode>> {
        let n_samples = indices.len();

        // Check stopping criteria
        if depth >= self.max_depth
            || n_samples < self.min_samples_split
            || self.is_pure(labels, indices)
        {
            return Ok(Box::new(TreeNode {
                feature_index: None,
                threshold: None,
                value: Some(self.compute_leaf_value(labels, indices)),
                left: None,
                right: None,
                samples: n_samples,
            }));
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) = self.find_best_split(data, labels, indices)?;

        // If no good split found, create leaf
        if best_gain < 1e-10 {
            return Ok(Box::new(TreeNode {
                feature_index: None,
                threshold: None,
                value: Some(self.compute_leaf_value(labels, indices)),
                left: None,
                right: None,
                samples: n_samples,
            }));
        }

        // Split data
        let (left_indices, right_indices) = self.split_data(data, indices, best_feature, best_threshold);

        // Check minimum samples per leaf
        if left_indices.len() < self.min_samples_leaf || right_indices.len() < self.min_samples_leaf {
            return Ok(Box::new(TreeNode {
                feature_index: None,
                threshold: None,
                value: Some(self.compute_leaf_value(labels, indices)),
                left: None,
                right: None,
                samples: n_samples,
            }));
        }

        // Recursively build subtrees
        let left = self.build_tree(data, labels, &left_indices, depth + 1)?;
        let right = self.build_tree(data, labels, &right_indices, depth + 1)?;

        Ok(Box::new(TreeNode {
            feature_index: Some(best_feature),
            threshold: Some(best_threshold),
            value: None,
            left: Some(left),
            right: Some(right),
            samples: n_samples,
        }))
    }

    fn find_best_split(
        &self,
        data: &[Vec<f64>],
        labels: &[f64],
        indices: &[usize],
    ) -> Result<(usize, f64, f64)> {
        let n_features = data[0].len();
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        let parent_impurity = self.compute_impurity(labels, indices);

        for feature in 0..n_features {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| data[i][feature]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            // Try each unique value as threshold
            for i in 0..values.len() - 1 {
                let threshold = f64::midpoint(values[i], values[i + 1]);

                let (left, right) = self.split_data(data, indices, feature, threshold);

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                // Compute information gain
                let left_impurity = self.compute_impurity(labels, &left);
                let right_impurity = self.compute_impurity(labels, &right);

                let n = indices.len() as f64;
                let n_left = left.len() as f64;
                let n_right = right.len() as f64;

                let gain = parent_impurity
                    - (n_left / n * left_impurity + n_right / n * right_impurity);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }

        Ok((best_feature, best_threshold, best_gain))
    }

    fn split_data(
        &self,
        data: &[Vec<f64>],
        indices: &[usize],
        feature: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for &idx in indices {
            if data[idx][feature] <= threshold {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }

        (left, right)
    }

    fn compute_impurity(&self, labels: &[f64], indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        if self.is_classifier {
            // Gini impurity for classification
            let mut counts: HashMap<i32, usize> = HashMap::new();
            for &idx in indices {
                *counts.entry(labels[idx] as i32).or_insert(0) += 1;
            }

            let n = indices.len() as f64;
            let mut gini = 1.0;
            for &count in counts.values() {
                let p = count as f64 / n;
                gini -= p * p;
            }
            gini
        } else {
            // Variance for regression
            let mean = indices.iter().map(|&i| labels[i]).sum::<f64>() / indices.len() as f64;
            indices
                .iter()
                .map(|&i| {
                    let diff = labels[i] - mean;
                    diff * diff
                })
                .sum::<f64>()
                / indices.len() as f64
        }
    }

    fn is_pure(&self, labels: &[f64], indices: &[usize]) -> bool {
        if indices.is_empty() {
            return true;
        }

        if self.is_classifier {
            let first_label = labels[indices[0]] as i32;
            indices.iter().all(|&i| labels[i] as i32 == first_label)
        } else {
            false // Regression is never "pure"
        }
    }

    fn compute_leaf_value(&self, labels: &[f64], indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        if self.is_classifier {
            // Most common class
            let mut counts: HashMap<i32, usize> = HashMap::new();
            for &idx in indices {
                *counts.entry(labels[idx] as i32).or_insert(0) += 1;
            }

            let (&class, _) = counts.iter().max_by_key(|(_, &count)| count).unwrap();
            f64::from(class)
        } else {
            // Mean value for regression
            indices.iter().map(|&i| labels[i]).sum::<f64>() / indices.len() as f64
        }
    }

    /// Predict single sample
    pub fn predict(&self, sample: &[f64]) -> Result<f64> {
        match &self.root {
            None => Err(Error::InvalidParameter(
                "Model not trained yet".to_string(),
            )),
            Some(root) => Ok(self.predict_node(root, sample)),
        }
    }

    fn predict_node(&self, node: &TreeNode, sample: &[f64]) -> f64 {
        if let Some(value) = node.value { value } else {
            let feature = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();

            if sample[feature] <= threshold {
                self.predict_node(node.left.as_ref().unwrap(), sample)
            } else {
                self.predict_node(node.right.as_ref().unwrap(), sample)
            }
        }
    }

    /// Get tree depth
    #[must_use] 
    pub fn get_depth(&self) -> usize {
        match &self.root {
            None => 0,
            Some(root) => self.node_depth(root),
        }
    }

    fn node_depth(&self, node: &TreeNode) -> usize {
        match (&node.left, &node.right) {
            (None, None) => 0, // Leaf has depth 0
            (Some(left), Some(right)) => {
                1 + self.node_depth(left).max(self.node_depth(right))
            }
            _ => 0, // Should not happen, but treat as leaf
        }
    }

    /// Get number of leaves
    #[must_use] 
    pub fn get_leaf_count(&self) -> usize {
        match &self.root {
            None => 0,
            Some(root) => self.count_leaves(root),
        }
    }

    fn count_leaves(&self, node: &TreeNode) -> usize {
        if node.value.is_some() {
            return 1;
        }

        let left_count = node
            .left
            .as_ref()
            .map_or(0, |n| self.count_leaves(n));
        let right_count = node
            .right
            .as_ref()
            .map_or(0, |n| self.count_leaves(n));

        left_count + right_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_classifier() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![6.0, 7.0],
            vec![7.0, 8.0],
            vec![8.0, 9.0],
        ];

        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut tree = DecisionTree::classifier().with_max_depth(5);
        tree.train(&data, &labels).unwrap();

        let prediction = tree.predict(&vec![1.5, 2.5]).unwrap();
        assert_eq!(prediction, 0.0);

        let prediction = tree.predict(&vec![7.5, 8.5]).unwrap();
        assert_eq!(prediction, 1.0);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let data = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];

        let labels = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut tree = DecisionTree::regressor().with_max_depth(5);
        tree.train(&data, &labels).unwrap();

        let prediction = tree.predict(&vec![2.5]).unwrap();
        assert!(prediction > 3.0 && prediction < 7.0);
    }

    #[test]
    fn test_tree_properties() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![6.0, 7.0],
        ];

        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let mut tree = DecisionTree::classifier().with_max_depth(3);
        tree.train(&data, &labels).unwrap();

        assert!(tree.get_depth() > 0);
        assert!(tree.get_leaf_count() >= 2);
    }
}
