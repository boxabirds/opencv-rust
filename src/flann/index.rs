use crate::flann::kdtree::KDTree;
use crate::flann::lsh::LSHIndex;
use crate::error::{Error, Result};

/// FLANN index for fast approximate nearest neighbor search
pub enum Index {
    KDTree(KDTree),
    LSH(LSHIndex),
    Linear(LinearIndex),
}

impl Index {
    /// Create KD-Tree index
    pub fn new_kdtree(data: &[Vec<f64>]) -> Result<Self> {
        Ok(Index::KDTree(KDTree::build(data)?))
    }

    /// Create LSH index
    pub fn new_lsh(dimension: usize, num_tables: usize, num_bits: usize) -> Self {
        Index::LSH(LSHIndex::new(dimension, num_tables, num_bits))
    }

    /// Create linear (brute-force) index
    pub fn new_linear(data: &[Vec<f64>]) -> Result<Self> {
        Ok(Index::Linear(LinearIndex::new(data)?))
    }

    /// Add data to index (only for LSH and Linear)
    pub fn add(&mut self, data: &[Vec<f64>]) -> Result<()> {
        match self {
            Index::LSH(lsh) => lsh.add(data),
            Index::Linear(linear) => linear.add(data),
            Index::KDTree(_) => Err(Error::InvalidParameter(
                "KDTree cannot add data after construction".to_string()
            )),
        }
    }

    /// K-nearest neighbor search
    pub fn knn_search(&self, query: &[f64], k: usize) -> Result<Vec<(usize, f64)>> {
        match self {
            Index::KDTree(kdtree) => kdtree.knn_search(query, k),
            Index::LSH(lsh) => lsh.knn_search(query, k),
            Index::Linear(linear) => linear.knn_search(query, k),
        }
    }

    /// Radius search
    pub fn radius_search(&self, query: &[f64], radius: f64) -> Result<Vec<(usize, f64)>> {
        match self {
            Index::KDTree(kdtree) => kdtree.radius_search(query, radius),
            Index::LSH(lsh) => lsh.radius_search(query, radius),
            Index::Linear(linear) => linear.radius_search(query, radius),
        }
    }
}

/// Linear (brute-force) index for exact nearest neighbor search
pub struct LinearIndex {
    data: Vec<Vec<f64>>,
    dimension: usize,
}

impl LinearIndex {
    pub fn new(data: &[Vec<f64>]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidParameter("Data cannot be empty".to_string()));
        }

        let dimension = data[0].len();

        for point in data {
            if point.len() != dimension {
                return Err(Error::InvalidDimensions(
                    "All points must have the same dimension".to_string()
                ));
            }
        }

        Ok(Self {
            data: data.to_vec(),
            dimension,
        })
    }

    pub fn add(&mut self, data: &[Vec<f64>]) -> Result<()> {
        for point in data {
            if point.len() != self.dimension {
                return Err(Error::InvalidDimensions(
                    format!("Expected dimension {}, got {}", self.dimension, point.len())
                ));
            }
        }

        self.data.extend_from_slice(data);
        Ok(())
    }

    pub fn knn_search(&self, query: &[f64], k: usize) -> Result<Vec<(usize, f64)>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidDimensions(
                format!("Query dimension {} doesn't match index dimension {}", query.len(), self.dimension)
            ));
        }

        let mut distances: Vec<(usize, f64)> = self.data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = euclidean_distance(query, point);
                (idx, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    pub fn radius_search(&self, query: &[f64], radius: f64) -> Result<Vec<(usize, f64)>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidDimensions(
                format!("Query dimension {} doesn't match index dimension {}", query.len(), self.dimension)
            ));
        }

        let results: Vec<(usize, f64)> = self.data
            .iter()
            .enumerate()
            .filter_map(|(idx, point)| {
                let dist = euclidean_distance(query, point);
                if dist <= radius {
                    Some((idx, dist))
                } else {
                    None
                }
            })
            .collect();

        Ok(results)
    }
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

/// Distance metrics for FLANN
#[derive(Debug, Clone, Copy)]
pub enum DistanceType {
    Euclidean,
    Manhattan,
    Minkowski(f64),
    ChiSquare,
    Hellinger,
    Hamming,
}

/// Index parameters for different index types
#[derive(Debug, Clone)]
pub struct IndexParams {
    pub algorithm: Algorithm,
    pub trees: usize,          // For KDTree
    pub branching: usize,      // For Hierarchical Clustering
    pub iterations: usize,     // For K-means
    pub centers_init: CentersInit,
    pub cb_index: f32,         // Cluster boundary index
}

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    Linear,
    KDTree,
    KMeans,
    Composite,
    LSH,
    Autotuned,
}

#[derive(Debug, Clone, Copy)]
pub enum CentersInit {
    Random,
    Gonzales,
    KMeansPP,
}

impl Default for IndexParams {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::KDTree,
            trees: 4,
            branching: 32,
            iterations: 11,
            centers_init: CentersInit::Random,
            cb_index: 0.2,
        }
    }
}

impl IndexParams {
    pub fn kdtree(trees: usize) -> Self {
        Self {
            algorithm: Algorithm::KDTree,
            trees,
            ..Default::default()
        }
    }

    pub fn lsh(table_number: usize, key_size: usize, multi_probe_level: usize) -> Self {
        Self {
            algorithm: Algorithm::LSH,
            trees: table_number,
            branching: key_size,
            iterations: multi_probe_level,
            ..Default::default()
        }
    }

    pub fn linear() -> Self {
        Self {
            algorithm: Algorithm::Linear,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_index() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
        ];

        let index = Index::new_linear(&data).unwrap();

        let query = vec![1.0, 1.0];
        let results = index.knn_search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Closest should be index 1
    }

    #[test]
    fn test_kdtree_index() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
        ];

        let index = Index::new_kdtree(&data).unwrap();

        let query = vec![1.0, 1.0];
        let results = index.knn_search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_lsh_index() {
        let mut index = Index::new_lsh(2, 5, 8);

        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![5.0, 5.0],
        ];

        index.add(&data).unwrap();

        let query = vec![0.0, 0.0];
        let results = index.knn_search(&query, 2).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_radius_search() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![5.0, 5.0],
        ];

        let index = Index::new_linear(&data).unwrap();

        let query = vec![0.0, 0.0];
        let results = index.radius_search(&query, 1.0).unwrap();

        assert!(results.len() >= 1);
        assert!(results.len() <= 2); // Should not include [5, 5]
    }
}
