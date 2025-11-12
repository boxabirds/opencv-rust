use crate::features2d::{Descriptor, hamming_distance};
use crate::error::{Error, Result};

/// Descriptor match between two keypoints
#[derive(Debug, Clone)]
pub struct DMatch {
    pub query_idx: usize,
    pub train_idx: usize,
    pub distance: f32,
}

impl DMatch {
    #[must_use] 
    pub fn new(query_idx: usize, train_idx: usize, distance: f32) -> Self {
        Self {
            query_idx,
            train_idx,
            distance,
        }
    }
}

/// Distance metrics for descriptor matching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceType {
    Hamming,
    L2,
}

/// Brute Force Matcher
pub struct BFMatcher {
    pub distance_type: DistanceType,
    pub cross_check: bool,
}

impl BFMatcher {
    #[must_use] 
    pub fn new(distance_type: DistanceType, cross_check: bool) -> Self {
        Self {
            distance_type,
            cross_check,
        }
    }

    /// Match descriptors from query set to train set
    pub fn match_descriptors(
        &self,
        query_desc: &[Descriptor],
        train_desc: &[Descriptor],
    ) -> Result<Vec<DMatch>> {
        if query_desc.is_empty() || train_desc.is_empty() {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();

        for (query_idx, q_desc) in query_desc.iter().enumerate() {
            let mut best_dist = f32::MAX;
            let mut best_idx = 0;

            for (train_idx, t_desc) in train_desc.iter().enumerate() {
                let dist = self.compute_distance(q_desc, t_desc)?;

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = train_idx;
                }
            }

            matches.push(DMatch::new(query_idx, best_idx, best_dist));
        }

        // Apply cross-check if enabled
        if self.cross_check {
            matches = self.apply_cross_check(query_desc, train_desc, matches)?;
        }

        Ok(matches)
    }

    /// K-nearest neighbors matching
    pub fn knn_match(
        &self,
        query_desc: &[Descriptor],
        train_desc: &[Descriptor],
        k: usize,
    ) -> Result<Vec<Vec<DMatch>>> {
        if query_desc.is_empty() || train_desc.is_empty() {
            return Ok(Vec::new());
        }

        let mut knn_matches = Vec::new();

        for (query_idx, q_desc) in query_desc.iter().enumerate() {
            let mut distances: Vec<(usize, f32)> = train_desc
                .iter()
                .enumerate()
                .map(|(train_idx, t_desc)| {
                    let dist = self.compute_distance(q_desc, t_desc).unwrap_or(f32::MAX);
                    (train_idx, dist)
                })
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Take k best matches
            let k_best: Vec<DMatch> = distances
                .iter()
                .take(k)
                .map(|&(train_idx, dist)| DMatch::new(query_idx, train_idx, dist))
                .collect();

            knn_matches.push(k_best);
        }

        Ok(knn_matches)
    }

    /// Radius matching - find all matches within a radius
    pub fn radius_match(
        &self,
        query_desc: &[Descriptor],
        train_desc: &[Descriptor],
        max_distance: f32,
    ) -> Result<Vec<Vec<DMatch>>> {
        if query_desc.is_empty() || train_desc.is_empty() {
            return Ok(Vec::new());
        }

        let mut radius_matches = Vec::new();

        for (query_idx, q_desc) in query_desc.iter().enumerate() {
            let mut matches_for_query = Vec::new();

            for (train_idx, t_desc) in train_desc.iter().enumerate() {
                let dist = self.compute_distance(q_desc, t_desc)?;

                if dist <= max_distance {
                    matches_for_query.push(DMatch::new(query_idx, train_idx, dist));
                }
            }

            // Sort by distance
            matches_for_query.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

            radius_matches.push(matches_for_query);
        }

        Ok(radius_matches)
    }

    fn compute_distance(&self, desc1: &[u8], desc2: &[u8]) -> Result<f32> {
        if desc1.len() != desc2.len() {
            return Err(Error::InvalidParameter(
                "Descriptors must have same length".to_string(),
            ));
        }

        let distance = match self.distance_type {
            DistanceType::Hamming => hamming_distance(desc1, desc2) as f32,
            DistanceType::L2 => {
                let sum: f64 = desc1
                    .iter()
                    .zip(desc2.iter())
                    .map(|(a, b)| {
                        let diff = f64::from(*a) - f64::from(*b);
                        diff * diff
                    })
                    .sum();
                sum.sqrt() as f32
            }
        };

        Ok(distance)
    }

    fn apply_cross_check(
        &self,
        query_desc: &[Descriptor],
        train_desc: &[Descriptor],
        forward_matches: Vec<DMatch>,
    ) -> Result<Vec<DMatch>> {
        // Compute backward matches
        let mut backward_best = vec![None; train_desc.len()];

        for (train_idx, t_desc) in train_desc.iter().enumerate() {
            let mut best_dist = f32::MAX;
            let mut best_query_idx = 0;

            for (query_idx, q_desc) in query_desc.iter().enumerate() {
                let dist = self.compute_distance(q_desc, t_desc)?;

                if dist < best_dist {
                    best_dist = dist;
                    best_query_idx = query_idx;
                }
            }

            backward_best[train_idx] = Some((best_query_idx, best_dist));
        }

        // Keep only bidirectional matches
        let mut cross_checked = Vec::new();

        for m in forward_matches {
            if let Some((best_query, _)) = backward_best[m.train_idx] {
                if best_query == m.query_idx {
                    cross_checked.push(m);
                }
            }
        }

        Ok(cross_checked)
    }
}

/// Apply Lowe's ratio test to filter matches
#[must_use] 
pub fn ratio_test_filter(knn_matches: &[Vec<DMatch>], ratio: f32) -> Vec<DMatch> {
    let mut good_matches = Vec::new();

    for matches in knn_matches {
        if matches.len() >= 2 {
            let best = &matches[0];
            let second_best = &matches[1];

            // Lowe's ratio test
            if best.distance < ratio * second_best.distance {
                good_matches.push(best.clone());
            }
        }
    }

    good_matches
}

/// Sort matches by distance
pub fn sort_matches_by_distance(matches: &mut [DMatch]) {
    matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf_matcher() {
        let query_desc = vec![vec![0b10101010, 0b11110000]];
        let train_desc = vec![
            vec![0b10101010, 0b11110000],
            vec![0b01010101, 0b00001111],
        ];

        let matcher = BFMatcher::new(DistanceType::Hamming, false);
        let matches = matcher.match_descriptors(&query_desc, &train_desc).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_idx, 0); // Should match first descriptor
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_knn_match() {
        let query_desc = vec![vec![0b10101010]];
        let train_desc = vec![vec![0b10101010], vec![0b10101011], vec![0b11111111]];

        let matcher = BFMatcher::new(DistanceType::Hamming, false);
        let knn = matcher.knn_match(&query_desc, &train_desc, 2).unwrap();

        assert_eq!(knn.len(), 1);
        assert_eq!(knn[0].len(), 2);
        assert!(knn[0][0].distance <= knn[0][1].distance);
    }

    #[test]
    fn test_ratio_test() {
        let knn_matches = vec![vec![
            DMatch::new(0, 0, 10.0),
            DMatch::new(0, 1, 30.0),
        ]];

        let good = ratio_test_filter(&knn_matches, 0.5);
        assert_eq!(good.len(), 1); // 10 < 0.5 * 30

        let bad = ratio_test_filter(&knn_matches, 0.3);
        assert_eq!(bad.len(), 0); // 10 >= 0.3 * 30
    }
}
