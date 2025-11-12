use crate::core::types::Point;
use crate::error::{Error, Result};
use crate::shape::moments::{Moments, hu_moments};

/// Methods for shape matching
#[derive(Clone, Copy)]
pub enum ShapeMatchMethod {
    I1,  // Sum of |1/mA - 1/mB|
    I2,  // Sum of |mA - mB|
    I3,  // Sum of |mA - mB| / |mA|
}

/// Compare two shapes using Hu moments
#[must_use] 
pub fn match_shapes(
    moments1: &Moments,
    moments2: &Moments,
    method: ShapeMatchMethod,
) -> f64 {
    let hu1 = hu_moments(moments1);
    let hu2 = hu_moments(moments2);

    match method {
        ShapeMatchMethod::I1 => {
            let mut sum = 0.0;
            for i in 0..7 {
                let ma = sign(hu1[i]) * hu1[i].abs().ln();
                let mb = sign(hu2[i]) * hu2[i].abs().ln();

                if ma.is_finite() && mb.is_finite() && ma != 0.0 && mb != 0.0 {
                    sum += (1.0 / ma - 1.0 / mb).abs();
                }
            }
            sum
        }
        ShapeMatchMethod::I2 => {
            let mut sum = 0.0;
            for i in 0..7 {
                let ma = sign(hu1[i]) * hu1[i].abs().ln();
                let mb = sign(hu2[i]) * hu2[i].abs().ln();

                if ma.is_finite() && mb.is_finite() {
                    sum += (ma - mb).abs();
                }
            }
            sum
        }
        ShapeMatchMethod::I3 => {
            let mut sum = 0.0;
            for i in 0..7 {
                let ma = sign(hu1[i]) * hu1[i].abs().ln();
                let mb = sign(hu2[i]) * hu2[i].abs().ln();

                if ma.is_finite() && mb.is_finite() && ma.abs() > 1e-10 {
                    sum += ((ma - mb) / ma).abs();
                }
            }
            sum
        }
    }
}

fn sign(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Hausdorff distance between two point sets
#[must_use] 
pub fn hausdorff_distance(contour1: &[Point], contour2: &[Point]) -> f64 {
    if contour1.is_empty() || contour2.is_empty() {
        return f64::INFINITY;
    }

    let h1 = directed_hausdorff(contour1, contour2);
    let h2 = directed_hausdorff(contour2, contour1);

    h1.max(h2)
}

fn directed_hausdorff(from: &[Point], to: &[Point]) -> f64 {
    let mut max_min_dist: f64 = 0.0;

    for p1 in from {
        let mut min_dist = f64::INFINITY;

        for p2 in to {
            let dx = f64::from(p1.x - p2.x);
            let dy = f64::from(p1.y - p2.y);
            let dist = (dx * dx + dy * dy).sqrt();

            min_dist = min_dist.min(dist);
        }

        max_min_dist = max_min_dist.max(min_dist);
    }

    max_min_dist
}

/// Frechet distance between two curves
pub fn frechet_distance(curve1: &[Point], curve2: &[Point]) -> Result<f64> {
    if curve1.is_empty() || curve2.is_empty() {
        return Err(Error::InvalidParameter("Empty curves".to_string()));
    }

    let n = curve1.len();
    let m = curve2.len();

    // Dynamic programming approach
    let mut ca = vec![vec![-1.0; m]; n];

    fn compute_ca(
        i: usize,
        j: usize,
        curve1: &[Point],
        curve2: &[Point],
        ca: &mut Vec<Vec<f64>>,
    ) -> f64 {
        if ca[i][j] > -1.0 {
            return ca[i][j];
        }

        let dist = point_distance(&curve1[i], &curve2[j]);

        ca[i][j] = if i == 0 && j == 0 {
            dist
        } else if i > 0 && j == 0 {
            compute_ca(i - 1, 0, curve1, curve2, ca).max(dist)
        } else if i == 0 && j > 0 {
            compute_ca(0, j - 1, curve1, curve2, ca).max(dist)
        } else {
            let c1 = compute_ca(i - 1, j, curve1, curve2, ca);
            let c2 = compute_ca(i - 1, j - 1, curve1, curve2, ca);
            let c3 = compute_ca(i, j - 1, curve1, curve2, ca);

            c1.min(c2).min(c3).max(dist)
        };

        ca[i][j]
    }

    Ok(compute_ca(n - 1, m - 1, curve1, curve2, &mut ca))
}

fn point_distance(p1: &Point, p2: &Point) -> f64 {
    let dx = f64::from(p1.x - p2.x);
    let dy = f64::from(p1.y - p2.y);
    (dx * dx + dy * dy).sqrt()
}

/// Shape context descriptor for a point in a contour
pub struct ShapeContext {
    pub bins_r: usize,
    pub bins_theta: usize,
    pub r_inner: f64,
    pub r_outer: f64,
}

impl Default for ShapeContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeContext {
    #[must_use] 
    pub fn new() -> Self {
        Self {
            bins_r: 5,
            bins_theta: 12,
            r_inner: 0.125,
            r_outer: 2.0,
        }
    }

    /// Compute shape context descriptor for a point
    #[must_use] 
    pub fn compute(&self, contour: &[Point], point_idx: usize) -> Vec<usize> {
        if point_idx >= contour.len() {
            return vec![0; self.bins_r * self.bins_theta];
        }

        let point = &contour[point_idx];
        let mut histogram = vec![0usize; self.bins_r * self.bins_theta];

        // Compute mean distance for normalization
        let mut mean_dist = 0.0;
        for p in contour {
            mean_dist += point_distance(point, p);
        }
        mean_dist /= contour.len() as f64;

        // Bin other points
        for (i, p) in contour.iter().enumerate() {
            if i == point_idx {
                continue;
            }

            let dx = f64::from(p.x - point.x);
            let dy = f64::from(p.y - point.y);

            let dist = (dx * dx + dy * dy).sqrt() / mean_dist;
            let angle = dy.atan2(dx);

            // Determine radial bin (log scale)
            let log_r = (dist / self.r_inner).ln();
            let log_r_outer = (self.r_outer / self.r_inner).ln();

            let r_bin = if dist < self.r_inner {
                0
            } else if dist > self.r_outer {
                self.bins_r - 1
            } else {
                ((log_r / log_r_outer) * (self.bins_r as f64)) as usize
            }.min(self.bins_r - 1);

            // Determine angular bin
            let theta_normalized = (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
            let theta_bin = (theta_normalized * self.bins_theta as f64) as usize % self.bins_theta;

            let bin_idx = r_bin * self.bins_theta + theta_bin;
            histogram[bin_idx] += 1;
        }

        histogram
    }

    /// Compute chi-square distance between two shape context histograms
    #[must_use] 
    pub fn chi_square_distance(&self, hist1: &[usize], hist2: &[usize]) -> f64 {
        if hist1.len() != hist2.len() {
            return f64::INFINITY;
        }

        let mut sum = 0.0;

        for (&h1, &h2) in hist1.iter().zip(hist2.iter()) {
            let num = f64::from((h1 as i32 - h2 as i32).pow(2));
            let denom = (h1 + h2) as f64;

            if denom > 0.0 {
                sum += num / denom;
            }
        }

        0.5 * sum
    }
}

/// Chamfer distance between two binary images (simplified version)
#[must_use] 
pub fn chamfer_distance(contour1: &[Point], contour2: &[Point]) -> f64 {
    if contour1.is_empty() || contour2.is_empty() {
        return f64::INFINITY;
    }

    let mut sum = 0.0;

    for p1 in contour1 {
        let mut min_dist = f64::INFINITY;

        for p2 in contour2 {
            let dist = point_distance(p1, p2);
            min_dist = min_dist.min(dist);
        }

        sum += min_dist;
    }

    sum / contour1.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::moments::Moments;

    #[test]
    fn test_match_shapes() {
        let mut m1 = Moments::new();
        m1.nu20 = 0.1;
        m1.nu11 = 0.05;
        m1.nu02 = 0.1;

        let mut m2 = Moments::new();
        m2.nu20 = 0.11;
        m2.nu11 = 0.05;
        m2.nu02 = 0.09;

        let dist = match_shapes(&m1, &m2, ShapeMatchMethod::I2);

        assert!(dist >= 0.0);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_hausdorff_distance() {
        let contour1 = vec![
            Point::new(0, 0),
            Point::new(1, 0),
            Point::new(2, 0),
        ];

        let contour2 = vec![
            Point::new(0, 0),
            Point::new(1, 0),
            Point::new(3, 0),
        ];

        let dist = hausdorff_distance(&contour1, &contour2);

        assert!(dist >= 0.0);
        assert!(dist < 10.0);
    }

    #[test]
    fn test_frechet_distance() {
        let curve1 = vec![
            Point::new(0, 0),
            Point::new(1, 0),
            Point::new(2, 0),
        ];

        let curve2 = vec![
            Point::new(0, 1),
            Point::new(1, 1),
            Point::new(2, 1),
        ];

        let dist = frechet_distance(&curve1, &curve2).unwrap();

        assert!(dist >= 0.0);
    }

    #[test]
    fn test_shape_context() {
        let contour = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let sc = ShapeContext::new();
        let descriptor = sc.compute(&contour, 0);

        assert_eq!(descriptor.len(), sc.bins_r * sc.bins_theta);
    }

    #[test]
    fn test_chamfer_distance() {
        let contour1 = vec![
            Point::new(0, 0),
            Point::new(10, 0),
        ];

        let contour2 = vec![
            Point::new(0, 1),
            Point::new(10, 1),
        ];

        let dist = chamfer_distance(&contour1, &contour2);

        assert!(dist >= 0.0);
        assert!((dist - 1.0).abs() < 0.1);
    }
}
