use crate::error::{Error, Result};
use std::collections::HashMap;

/// K-Nearest Neighbors classifier/regressor
pub struct KNearest {
    k: usize,
    data: Vec<Vec<f64>>,
    labels: Vec<f64>,
    algorithm: Algorithm,
    is_classifier: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    /// Brute force search
    BruteForce,
    /// KD-Tree (faster for low dimensions)
    KDTree,
}

impl KNearest {
    /// Create new KNN classifier
    #[must_use] 
    pub fn classifier(k: usize) -> Self {
        Self {
            k,
            data: Vec::new(),
            labels: Vec::new(),
            algorithm: Algorithm::BruteForce,
            is_classifier: true,
        }
    }

    /// Create new KNN regressor
    #[must_use] 
    pub fn regressor(k: usize) -> Self {
        Self {
            k,
            data: Vec::new(),
            labels: Vec::new(),
            algorithm: Algorithm::BruteForce,
            is_classifier: false,
        }
    }

    /// Set algorithm
    #[must_use] 
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set k value
    #[must_use] 
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Train (store) the data
    pub fn train(&mut self, data: &[Vec<f64>], labels: &[f64]) -> Result<()> {
        if data.len() != labels.len() {
            return Err(Error::InvalidParameter(
                "Data and labels must have same length".to_string(),
            ));
        }

        if data.is_empty() {
            return Err(Error::InvalidParameter("Data cannot be empty".to_string()));
        }

        if self.k > data.len() {
            return Err(Error::InvalidParameter(
                format!("k ({}) cannot be greater than number of samples ({})", self.k, data.len()),
            ));
        }

        self.data = data.to_vec();
        self.labels = labels.to_vec();

        Ok(())
    }

    /// Predict single sample
    pub fn predict(&self, sample: &[f64]) -> Result<f64> {
        if self.data.is_empty() {
            return Err(Error::InvalidParameter("Model not trained yet".to_string()));
        }

        // Find k nearest neighbors
        let neighbors = self.find_knearest(sample)?;

        if self.is_classifier {
            // Majority voting
            let mut votes: HashMap<i32, f64> = HashMap::new();

            for (label, distance) in neighbors {
                // Weighted voting by inverse distance
                let weight = if distance > 0.0 { 1.0 / distance } else { 1e6 };
                *votes.entry(label as i32).or_insert(0.0) += weight;
            }

            let (&class, _) = votes
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            Ok(f64::from(class))
        } else {
            // Weighted average for regression
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (label, distance) in neighbors {
                let weight = if distance > 0.0 { 1.0 / distance } else { 1e6 };
                weighted_sum += label * weight;
                weight_sum += weight;
            }

            Ok(weighted_sum / weight_sum)
        }
    }

    fn find_knearest(&self, sample: &[f64]) -> Result<Vec<(f64, f64)>> {
        match self.algorithm {
            Algorithm::BruteForce => self.brute_force_search(sample),
            Algorithm::KDTree => {
                // For simplicity, fall back to brute force
                // Real implementation would build and query KD-tree
                self.brute_force_search(sample)
            }
        }
    }

    fn brute_force_search(&self, sample: &[f64]) -> Result<Vec<(f64, f64)>> {
        // Compute distances to all points
        let mut distances: Vec<(f64, f64, usize)> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = euclidean_distance(sample, point);
                (dist, self.labels[idx], idx)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Return k nearest
        Ok(distances
            .into_iter()
            .take(self.k)
            .map(|(dist, label, _)| (label, dist))
            .collect())
    }

    /// Find k nearest neighbors and return their indices and distances
    pub fn find_nearest(&self, sample: &[f64], k: usize) -> Result<Vec<(usize, f64)>> {
        if self.data.is_empty() {
            return Err(Error::InvalidParameter("Model not trained yet".to_string()));
        }

        let mut distances: Vec<(usize, f64)> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, point)| (idx, euclidean_distance(sample, point)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(distances.into_iter().take(k).collect())
    }

    /// Predict with different distance metrics
    pub fn predict_with_distance(&self, sample: &[f64], distance_fn: DistanceMetric) -> Result<f64> {
        if self.data.is_empty() {
            return Err(Error::InvalidParameter("Model not trained yet".to_string()));
        }

        let distance_func: fn(&[f64], &[f64]) -> f64 = match distance_fn {
            DistanceMetric::Euclidean => euclidean_distance,
            DistanceMetric::Manhattan => manhattan_distance,
            DistanceMetric::Chebyshev => chebyshev_distance,
            DistanceMetric::Minkowski(p) => {
                return self.predict_with_minkowski(sample, p);
            }
        };

        let mut distances: Vec<(f64, f64)> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = distance_func(sample, point);
                (dist, self.labels[idx])
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let neighbors: Vec<(f64, f64)> = distances
            .into_iter()
            .take(self.k)
            .map(|(dist, label)| (label, dist))
            .collect();

        if self.is_classifier {
            let mut votes: HashMap<i32, f64> = HashMap::new();
            for (label, distance) in neighbors {
                let weight = if distance > 0.0 { 1.0 / distance } else { 1e6 };
                *votes.entry(label as i32).or_insert(0.0) += weight;
            }

            let (&class, _) = votes
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            Ok(f64::from(class))
        } else {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (label, distance) in neighbors {
                let weight = if distance > 0.0 { 1.0 / distance } else { 1e6 };
                weighted_sum += label * weight;
                weight_sum += weight;
            }

            Ok(weighted_sum / weight_sum)
        }
    }

    fn predict_with_minkowski(&self, sample: &[f64], p: f64) -> Result<f64> {
        let mut distances: Vec<(f64, f64)> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = minkowski_distance(sample, point, p);
                (dist, self.labels[idx])
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let neighbors: Vec<(f64, f64)> = distances
            .into_iter()
            .take(self.k)
            .map(|(dist, label)| (label, dist))
            .collect();

        if self.is_classifier {
            let mut votes: HashMap<i32, usize> = HashMap::new();
            for (label, _) in neighbors {
                *votes.entry(label as i32).or_insert(0) += 1;
            }

            let (&class, _) = votes.iter().max_by_key(|(_, &count)| count).unwrap();
            Ok(f64::from(class))
        } else {
            let sum: f64 = neighbors.iter().map(|(label, _)| label).sum();
            Ok(sum / neighbors.len() as f64)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Minkowski(f64),
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_classifier() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![6.0, 7.0],
            vec![7.0, 8.0],
            vec![8.0, 9.0],
        ];

        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut knn = KNearest::classifier(3);
        knn.train(&data, &labels).unwrap();

        let prediction = knn.predict(&vec![2.0, 3.0]).unwrap();
        assert_eq!(prediction, 0.0);

        let prediction = knn.predict(&vec![7.0, 8.0]).unwrap();
        assert_eq!(prediction, 1.0);
    }

    #[test]
    fn test_knn_regressor() {
        let data = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];

        let labels = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut knn = KNearest::regressor(3);
        knn.train(&data, &labels).unwrap();

        let prediction = knn.predict(&vec![3.0]).unwrap();
        assert!(prediction > 4.0 && prediction < 8.0);
    }

    #[test]
    fn test_find_nearest() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ];

        let labels = vec![0.0, 1.0, 2.0];

        let mut knn = KNearest::classifier(2);
        knn.train(&data, &labels).unwrap();

        let nearest = knn.find_nearest(&vec![1.5, 1.5], 2).unwrap();
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, 1); // Index 1 is closest
    }

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let euc = euclidean_distance(&a, &b);
        let man = manhattan_distance(&a, &b);
        let cheb = chebyshev_distance(&a, &b);

        assert!(euc > 0.0);
        assert!(man > 0.0);
        assert!(cheb > 0.0);
        assert!(man > euc); // Manhattan >= Euclidean
    }
}
