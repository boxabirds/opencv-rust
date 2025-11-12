use crate::error::{Error, Result};
use std::collections::HashMap;

/// Locality Sensitive Hashing index for approximate nearest neighbor search
pub struct LSHIndex {
    data: Vec<Vec<f64>>,
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
    projections: Vec<Vec<Vec<f64>>>,
    num_tables: usize,
    num_bits: usize,
    dimension: usize,
}

impl LSHIndex {
    /// Create new LSH index
    #[must_use] 
    pub fn new(dimension: usize, num_tables: usize, num_bits: usize) -> Self {
        Self {
            data: Vec::new(),
            hash_tables: vec![HashMap::new(); num_tables],
            projections: Vec::new(),
            num_tables,
            num_bits,
            dimension,
        }
    }

    /// Add points to the index
    pub fn add(&mut self, data: &[Vec<f64>]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        // Verify dimensions
        for point in data {
            if point.len() != self.dimension {
                return Err(Error::InvalidDimensions(
                    format!("Expected dimension {}, got {}", self.dimension, point.len())
                ));
            }
        }

        // Generate random projections if not done yet
        if self.projections.is_empty() {
            self.generate_random_projections();
        }

        let start_idx = self.data.len();
        self.data.extend_from_slice(data);

        // Hash each point into all tables
        for table_idx in 0..self.num_tables {
            for (point_offset, point) in data.iter().enumerate() {
                let hash = self.hash_point(point, table_idx);
                self.hash_tables[table_idx].entry(hash)
                    .or_default()
                    .push(start_idx + point_offset);
            }
        }

        Ok(())
    }

    fn generate_random_projections(&mut self) {
        use std::f64::consts::PI;

        self.projections = Vec::with_capacity(self.num_tables);

        for table_idx in 0..self.num_tables {
            let mut table_projections = Vec::with_capacity(self.num_bits);

            for bit_idx in 0..self.num_bits {
                let mut projection = Vec::with_capacity(self.dimension);

                // Generate random unit vector (simplified - using sine/cosine)
                for dim_idx in 0..self.dimension {
                    let seed = (table_idx * 1000 + bit_idx * 100 + dim_idx) as f64;
                    let angle = seed * PI / 180.0;
                    projection.push(angle.sin());
                }

                // Normalize
                let norm: f64 = projection.iter().map(|x| x * x).sum::<f64>().sqrt();
                for val in &mut projection {
                    *val /= norm;
                }

                table_projections.push(projection);
            }

            self.projections.push(table_projections);
        }
    }

    fn hash_point(&self, point: &[f64], table_idx: usize) -> u64 {
        let mut hash = 0u64;

        for (bit_idx, projection) in self.projections[table_idx].iter().enumerate() {
            // Dot product
            let dot: f64 = point.iter()
                .zip(projection.iter())
                .map(|(a, b)| a * b)
                .sum();

            if dot > 0.0 {
                hash |= 1 << bit_idx;
            }
        }

        hash
    }

    /// Find approximate k nearest neighbors
    pub fn knn_search(&self, query: &[f64], k: usize) -> Result<Vec<(usize, f64)>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidDimensions(
                format!("Query dimension {} doesn't match index dimension {}", query.len(), self.dimension)
            ));
        }

        // Collect candidates from all hash tables
        let mut candidates = HashMap::new();

        for table_idx in 0..self.num_tables {
            let hash = self.hash_point(query, table_idx);

            if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                for &idx in bucket {
                    *candidates.entry(idx).or_insert(0) += 1;
                }
            }
        }

        // Compute exact distances for candidates
        let mut distances: Vec<(usize, f64)> = candidates
            .keys()
            .map(|&idx| {
                let dist = euclidean_distance(query, &self.data[idx]);
                (idx, dist)
            })
            .collect();

        // Sort by distance and return top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    /// Find all points within radius
    pub fn radius_search(&self, query: &[f64], radius: f64) -> Result<Vec<(usize, f64)>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidDimensions(
                format!("Query dimension {} doesn't match index dimension {}", query.len(), self.dimension)
            ));
        }

        // Collect candidates
        let mut candidates = HashMap::new();

        for table_idx in 0..self.num_tables {
            let hash = self.hash_point(query, table_idx);

            if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                for &idx in bucket {
                    *candidates.entry(idx).or_insert(0) += 1;
                }
            }
        }

        // Filter by radius
        let results: Vec<(usize, f64)> = candidates
            .keys()
            .filter_map(|&idx| {
                let dist = euclidean_distance(query, &self.data[idx]);
                if dist <= radius {
                    Some((idx, dist))
                } else {
                    None
                }
            })
            .collect();

        Ok(results)
    }

    /// Get number of indexed points
    #[must_use] 
    pub fn size(&self) -> usize {
        self.data.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_creation() {
        let mut index = LSHIndex::new(2, 5, 8);

        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ];

        index.add(&data).unwrap();
        assert_eq!(index.size(), 3);
    }

    #[test]
    fn test_lsh_knn_search() {
        let mut index = LSHIndex::new(2, 10, 16);

        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
        ];

        index.add(&data).unwrap();

        let query = vec![0.0, 0.0];
        let results = index.knn_search(&query, 2).unwrap();

        // Should find points near [0, 0]
        assert!(!results.is_empty());
        assert!(results[0].1 < 1.0); // First result should be close
    }

    #[test]
    fn test_lsh_radius_search() {
        let mut index = LSHIndex::new(2, 10, 16);

        let data = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![5.0, 5.0],
        ];

        index.add(&data).unwrap();

        let query = vec![0.0, 0.0];
        let results = index.radius_search(&query, 1.0).unwrap();

        // Should find points within radius 1.0
        assert!(results.len() >= 1);
    }
}
