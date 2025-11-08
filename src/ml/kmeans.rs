use crate::error::{Error, Result};
use std::f64;

/// K-means clustering flags
#[derive(Debug, Clone, Copy)]
pub enum KMeansFlags {
    RandomCenters,
    PPCenters,
    UseInitialLabels,
}

/// Perform k-means clustering
pub fn kmeans(
    data: &[Vec<f64>],
    k: usize,
    labels: &mut [i32],
    max_iter: i32,
    epsilon: f64,
    flags: KMeansFlags,
) -> Result<(Vec<Vec<f64>>, f64)> {
    if data.is_empty() {
        return Err(Error::InvalidParameter("Data is empty".to_string()));
    }

    if k == 0 || k > data.len() {
        return Err(Error::InvalidParameter(
            "Invalid number of clusters".to_string(),
        ));
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Initialize centers
    let mut centers = match flags {
        KMeansFlags::RandomCenters => initialize_random_centers(data, k),
        KMeansFlags::PPCenters => initialize_pp_centers(data, k),
        KMeansFlags::UseInitialLabels => initialize_from_labels(data, labels, k)?,
    };

    let mut prev_compactness = f64::MAX;

    for _iter in 0..max_iter {
        // Assignment step: assign each point to nearest center
        for (i, point) in data.iter().enumerate() {
            let mut min_dist = f64::MAX;
            let mut best_cluster = 0;

            for (j, center) in centers.iter().enumerate() {
                let dist = euclidean_distance(point, center);

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster as i32;
        }

        // Update step: recalculate centers
        let mut new_centers = vec![vec![0.0; n_features]; k];
        let mut counts = vec![0; k];

        for (i, point) in data.iter().enumerate() {
            let cluster = labels[i] as usize;

            for (j, &val) in point.iter().enumerate() {
                new_centers[cluster][j] += val;
            }

            counts[cluster] += 1;
        }

        for (i, center) in new_centers.iter_mut().enumerate() {
            if counts[i] > 0 {
                for val in center.iter_mut() {
                    *val /= counts[i] as f64;
                }
            }
        }

        // Calculate compactness (within-cluster sum of squares)
        let compactness = calculate_compactness(data, labels, &new_centers);

        // Check convergence
        if (prev_compactness - compactness).abs() < epsilon {
            centers = new_centers;
            break;
        }

        centers = new_centers;
        prev_compactness = compactness;
    }

    let final_compactness = calculate_compactness(data, labels, &centers);

    Ok((centers, final_compactness))
}

fn initialize_random_centers(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut centers = Vec::new();

    // Simple pseudo-random selection
    let step = data.len() / k;

    for i in 0..k {
        let idx = (i * step).min(data.len() - 1);
        centers.push(data[idx].clone());
    }

    centers
}

fn initialize_pp_centers(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    // K-means++ initialization
    let mut centers = Vec::new();

    // Choose first center randomly
    centers.push(data[0].clone());

    // Choose remaining centers
    for _ in 1..k {
        let mut distances = Vec::new();

        for point in data {
            let mut min_dist = f64::MAX;

            for center in &centers {
                let dist = euclidean_distance(point, center);
                min_dist = min_dist.min(dist);
            }

            distances.push(min_dist * min_dist);
        }

        // Choose point with maximum distance
        let max_idx = distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        centers.push(data[max_idx].clone());
    }

    centers
}

fn initialize_from_labels(
    data: &[Vec<f64>],
    labels: &[i32],
    k: usize,
) -> Result<Vec<Vec<f64>>> {
    let n_features = data[0].len();
    let mut centers = vec![vec![0.0; n_features]; k];
    let mut counts = vec![0; k];

    for (i, point) in data.iter().enumerate() {
        let cluster = labels[i] as usize;

        if cluster >= k {
            return Err(Error::InvalidParameter(
                "Invalid label in initial labels".to_string(),
            ));
        }

        for (j, &val) in point.iter().enumerate() {
            centers[cluster][j] += val;
        }

        counts[cluster] += 1;
    }

    for (i, center) in centers.iter_mut().enumerate() {
        if counts[i] > 0 {
            for val in center.iter_mut() {
                *val /= counts[i] as f64;
            }
        }
    }

    Ok(centers)
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn calculate_compactness(
    data: &[Vec<f64>],
    labels: &[i32],
    centers: &[Vec<f64>],
) -> f64 {
    let mut compactness = 0.0;

    for (i, point) in data.iter().enumerate() {
        let cluster = labels[i] as usize;
        let dist = euclidean_distance(point, &centers[cluster]);
        compactness += dist * dist;
    }

    compactness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans() {
        let data = vec![
            vec![1.0, 1.0],
            vec![1.5, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 7.0],
            vec![3.5, 5.0],
            vec![4.5, 5.0],
        ];

        let mut labels = vec![0; data.len()];

        let (centers, compactness) =
            kmeans(&data, 2, &mut labels, 100, 1.0, KMeansFlags::PPCenters).unwrap();

        assert_eq!(centers.len(), 2);
        assert!(compactness >= 0.0);
    }
}
