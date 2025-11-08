use crate::error::{Error, Result};

/// KD-Tree for fast nearest neighbor search
pub struct KDTree {
    root: Option<Box<KDNode>>,
    dimension: usize,
    data: Vec<Vec<f64>>,
}

struct KDNode {
    point_idx: usize,
    split_dim: usize,
    split_value: f64,
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
}

impl KDTree {
    /// Build KD-Tree from data points
    pub fn build(data: &[Vec<f64>]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidParameter("Data cannot be empty".to_string()));
        }

        let dimension = data[0].len();

        // Verify all points have same dimension
        for point in data {
            if point.len() != dimension {
                return Err(Error::InvalidDimensions(
                    "All points must have the same dimension".to_string()
                ));
            }
        }

        let indices: Vec<usize> = (0..data.len()).collect();

        let root = Self::build_recursive(data, &indices, 0, dimension)?;

        Ok(Self {
            root: Some(root),
            dimension,
            data: data.to_vec(),
        })
    }

    fn build_recursive(
        data: &[Vec<f64>],
        indices: &[usize],
        depth: usize,
        dimension: usize,
    ) -> Result<Box<KDNode>> {
        if indices.is_empty() {
            return Err(Error::InvalidParameter("Empty indices".to_string()));
        }

        // Choose split dimension (cycle through dimensions)
        let split_dim = depth % dimension;

        // Sort indices by split dimension
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            data[a][split_dim].partial_cmp(&data[b][split_dim]).unwrap()
        });

        // Choose median
        let median_idx = sorted_indices.len() / 2;
        let point_idx = sorted_indices[median_idx];
        let split_value = data[point_idx][split_dim];

        // Build left and right subtrees
        let left = if median_idx > 0 {
            Some(Self::build_recursive(
                data,
                &sorted_indices[..median_idx],
                depth + 1,
                dimension,
            )?)
        } else {
            None
        };

        let right = if median_idx < sorted_indices.len() - 1 {
            Some(Self::build_recursive(
                data,
                &sorted_indices[median_idx + 1..],
                depth + 1,
                dimension,
            )?)
        } else {
            None
        };

        Ok(Box::new(KDNode {
            point_idx,
            split_dim,
            split_value,
            left,
            right,
        }))
    }

    /// Find k nearest neighbors
    pub fn knn_search(&self, query: &[f64], k: usize) -> Result<Vec<(usize, f64)>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidDimensions(
                format!("Query dimension {} doesn't match tree dimension {}", query.len(), self.dimension)
            ));
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        let mut heap = Vec::new(); // (distance, index)

        if let Some(ref root) = self.root {
            self.search_recursive(root, query, k, &mut heap)?;
        }

        // Sort by distance
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(heap)
    }

    fn search_recursive(
        &self,
        node: &KDNode,
        query: &[f64],
        k: usize,
        heap: &mut Vec<(usize, f64)>,
    ) -> Result<()> {
        // Compute distance to current point
        let dist = euclidean_distance(query, &self.data[node.point_idx]);

        // Add to heap if we have room or if this is closer
        if heap.len() < k {
            heap.push((node.point_idx, dist));
            if heap.len() == k {
                heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Max heap
            }
        } else if dist < heap[0].1 {
            heap[0] = (node.point_idx, dist);
            heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }

        // Determine which side to search first
        let diff = query[node.split_dim] - node.split_value;
        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search closer side first
        if let Some(ref child) = first {
            self.search_recursive(child, query, k, heap)?;
        }

        // Check if we need to search the other side
        let split_dist = diff.abs();
        if heap.len() < k || split_dist < heap[0].1 {
            if let Some(ref child) = second {
                self.search_recursive(child, query, k, heap)?;
            }
        }

        Ok(())
    }

    /// Radius search - find all points within radius
    pub fn radius_search(&self, query: &[f64], radius: f64) -> Result<Vec<(usize, f64)>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidDimensions(
                format!("Query dimension {} doesn't match tree dimension {}", query.len(), self.dimension)
            ));
        }

        let mut results = Vec::new();

        if let Some(ref root) = self.root {
            self.radius_search_recursive(root, query, radius, &mut results)?;
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(results)
    }

    fn radius_search_recursive(
        &self,
        node: &KDNode,
        query: &[f64],
        radius: f64,
        results: &mut Vec<(usize, f64)>,
    ) -> Result<()> {
        let dist = euclidean_distance(query, &self.data[node.point_idx]);

        if dist <= radius {
            results.push((node.point_idx, dist));
        }

        let diff = query[node.split_dim] - node.split_value;

        // Search both sides if necessary
        if diff < 0.0 {
            if let Some(ref left) = node.left {
                self.radius_search_recursive(left, query, radius, results)?;
            }
            if diff.abs() <= radius {
                if let Some(ref right) = node.right {
                    self.radius_search_recursive(right, query, radius, results)?;
                }
            }
        } else {
            if let Some(ref right) = node.right {
                self.radius_search_recursive(right, query, radius, results)?;
            }
            if diff.abs() <= radius {
                if let Some(ref left) = node.left {
                    self.radius_search_recursive(left, query, radius, results)?;
                }
            }
        }

        Ok(())
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
    fn test_kdtree_build() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];

        let tree = KDTree::build(&data).unwrap();
        assert_eq!(tree.dimension, 2);
    }

    #[test]
    fn test_knn_search() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
        ];

        let tree = KDTree::build(&data).unwrap();

        let query = vec![1.5, 1.5];
        let results = tree.knn_search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // Closest should be [1, 1] or [2, 2]
        assert!(results[0].0 == 1 || results[0].0 == 2);
    }

    #[test]
    fn test_radius_search() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];

        let tree = KDTree::build(&data).unwrap();

        let query = vec![0.0, 0.0];
        let results = tree.radius_search(&query, 1.5).unwrap();

        // Should find [0,0], [1,0], [0,1] but not [5,5]
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert_eq!(dist, 5.0);
    }
}
