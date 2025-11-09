use crate::core::Mat;
use crate::error::{Error, Result};
use crate::core::types::Point;

/// Image moments up to 3rd order
#[derive(Debug, Clone)]
pub struct Moments {
    // Spatial moments
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m11: f64,
    pub m02: f64,
    pub m30: f64,
    pub m21: f64,
    pub m12: f64,
    pub m03: f64,

    // Central moments
    pub mu20: f64,
    pub mu11: f64,
    pub mu02: f64,
    pub mu30: f64,
    pub mu21: f64,
    pub mu12: f64,
    pub mu03: f64,

    // Central normalized moments
    pub nu20: f64,
    pub nu11: f64,
    pub nu02: f64,
    pub nu30: f64,
    pub nu21: f64,
    pub nu12: f64,
    pub nu03: f64,
}

impl Moments {
    pub fn new() -> Self {
        Self {
            m00: 0.0,
            m10: 0.0,
            m01: 0.0,
            m20: 0.0,
            m11: 0.0,
            m02: 0.0,
            m30: 0.0,
            m21: 0.0,
            m12: 0.0,
            m03: 0.0,
            mu20: 0.0,
            mu11: 0.0,
            mu02: 0.0,
            mu30: 0.0,
            mu21: 0.0,
            mu12: 0.0,
            mu03: 0.0,
            nu20: 0.0,
            nu11: 0.0,
            nu02: 0.0,
            nu30: 0.0,
            nu21: 0.0,
            nu12: 0.0,
            nu03: 0.0,
        }
    }

    /// Compute centroid from spatial moments
    pub fn centroid(&self) -> (f64, f64) {
        if self.m00 != 0.0 {
            (self.m10 / self.m00, self.m01 / self.m00)
        } else {
            (0.0, 0.0)
        }
    }

    /// Compute area (for binary images, m00 is the area)
    pub fn area(&self) -> f64 {
        self.m00
    }

    /// Compute orientation angle
    pub fn orientation(&self) -> f64 {
        if self.mu20 - self.mu02 == 0.0 && self.mu11 == 0.0 {
            0.0
        } else {
            0.5 * (2.0 * self.mu11).atan2(self.mu20 - self.mu02)
        }
    }

    /// Compute eccentricity
    pub fn eccentricity(&self) -> f64 {
        let a = self.mu20 + self.mu02;
        let b = ((self.mu20 - self.mu02).powi(2) + 4.0 * self.mu11.powi(2)).sqrt();

        let lambda1 = 0.5 * (a + b);
        let lambda2 = 0.5 * (a - b);

        if lambda1 > 0.0 {
            (1.0 - lambda2 / lambda1).sqrt()
        } else {
            0.0
        }
    }
}

/// Compute moments for a binary or grayscale image
pub fn compute_moments(image: &Mat) -> Result<Moments> {
    if image.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Moments require single-channel image".to_string(),
        ));
    }

    let mut moments = Moments::new();

    // Compute spatial moments
    for row in 0..image.rows() {
        for col in 0..image.cols() {
            let intensity = image.at(row, col)?[0] as f64;

            if intensity > 0.0 {
                let x = col as f64;
                let y = row as f64;

                // Spatial moments
                moments.m00 += intensity;
                moments.m10 += x * intensity;
                moments.m01 += y * intensity;
                moments.m20 += x * x * intensity;
                moments.m11 += x * y * intensity;
                moments.m02 += y * y * intensity;
                moments.m30 += x * x * x * intensity;
                moments.m21 += x * x * y * intensity;
                moments.m12 += x * y * y * intensity;
                moments.m03 += y * y * y * intensity;
            }
        }
    }

    // Compute central moments
    if moments.m00 != 0.0 {
        let x_bar = moments.m10 / moments.m00;
        let y_bar = moments.m01 / moments.m00;

        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let intensity = image.at(row, col)?[0] as f64;

                if intensity > 0.0 {
                    let x = col as f64 - x_bar;
                    let y = row as f64 - y_bar;

                    moments.mu20 += x * x * intensity;
                    moments.mu11 += x * y * intensity;
                    moments.mu02 += y * y * intensity;
                    moments.mu30 += x * x * x * intensity;
                    moments.mu21 += x * x * y * intensity;
                    moments.mu12 += x * y * y * intensity;
                    moments.mu03 += y * y * y * intensity;
                }
            }
        }

        // Compute normalized central moments
        let m00_pow = moments.m00.powf(2.5);

        if m00_pow != 0.0 {
            moments.nu20 = moments.mu20 / moments.m00.powf(2.0);
            moments.nu11 = moments.mu11 / moments.m00.powf(2.0);
            moments.nu02 = moments.mu02 / moments.m00.powf(2.0);
            moments.nu30 = moments.mu30 / m00_pow;
            moments.nu21 = moments.mu21 / m00_pow;
            moments.nu12 = moments.mu12 / m00_pow;
            moments.nu03 = moments.mu03 / m00_pow;
        }
    }

    Ok(moments)
}

/// Compute Hu's 7 invariant moments
pub fn hu_moments(moments: &Moments) -> [f64; 7] {
    let nu20 = moments.nu20;
    let nu11 = moments.nu11;
    let nu02 = moments.nu02;
    let nu30 = moments.nu30;
    let nu21 = moments.nu21;
    let nu12 = moments.nu12;
    let nu03 = moments.nu03;

    let hu1 = nu20 + nu02;

    let hu2 = (nu20 - nu02).powi(2) + 4.0 * nu11.powi(2);

    let hu3 = (nu30 - 3.0 * nu12).powi(2) + (3.0 * nu21 - nu03).powi(2);

    let hu4 = (nu30 + nu12).powi(2) + (nu21 + nu03).powi(2);

    let hu5 = (nu30 - 3.0 * nu12) * (nu30 + nu12)
        * ((nu30 + nu12).powi(2) - 3.0 * (nu21 + nu03).powi(2))
        + (3.0 * nu21 - nu03) * (nu21 + nu03)
        * (3.0 * (nu30 + nu12).powi(2) - (nu21 + nu03).powi(2));

    let hu6 = (nu20 - nu02)
        * ((nu30 + nu12).powi(2) - (nu21 + nu03).powi(2))
        + 4.0 * nu11 * (nu30 + nu12) * (nu21 + nu03);

    let hu7 = (3.0 * nu21 - nu03) * (nu30 + nu12)
        * ((nu30 + nu12).powi(2) - 3.0 * (nu21 + nu03).powi(2))
        - (nu30 - 3.0 * nu12) * (nu21 + nu03)
        * (3.0 * (nu30 + nu12).powi(2) - (nu21 + nu03).powi(2));

    [hu1, hu2, hu3, hu4, hu5, hu6, hu7]
}

/// Compute contour moments from a set of points
pub fn contour_moments(contour: &[Point]) -> Moments {
    let mut moments = Moments::new();

    let n = contour.len();
    if n < 3 {
        return moments;
    }

    // Using Green's theorem for contour integrals
    for i in 0..n {
        let curr = &contour[i];
        let next = &contour[(i + 1) % n];

        let xi = curr.x as f64;
        let yi = curr.y as f64;
        let xi1 = next.x as f64;
        let yi1 = next.y as f64;

        let a = xi * yi1 - xi1 * yi;

        moments.m00 += a;
        moments.m10 += (xi + xi1) * a;
        moments.m01 += (yi + yi1) * a;
        moments.m20 += (xi * xi + xi * xi1 + xi1 * xi1) * a;
        moments.m11 += (xi * yi + 0.5 * (xi * yi1 + xi1 * yi) + xi1 * yi1) * a;
        moments.m02 += (yi * yi + yi * yi1 + yi1 * yi1) * a;
    }

    moments.m00 *= 0.5;
    moments.m10 /= 6.0;
    moments.m01 /= 6.0;
    moments.m20 /= 12.0;
    moments.m11 /= 24.0;
    moments.m02 /= 12.0;

    // Compute central moments
    if moments.m00.abs() > 1e-10 {
        let x_bar = moments.m10 / moments.m00;
        let y_bar = moments.m01 / moments.m00;

        moments.mu20 = moments.m20 - x_bar * moments.m10;
        moments.mu11 = moments.m11 - x_bar * moments.m01;
        moments.mu02 = moments.m02 - y_bar * moments.m01;

        // Normalized moments
        let norm = moments.m00.powf(2.0);
        if norm != 0.0 {
            moments.nu20 = moments.mu20 / norm;
            moments.nu11 = moments.mu11 / norm;
            moments.nu02 = moments.mu02 / norm;
        }
    }

    moments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MatDepth;
    use crate::core::types::Scalar;

    #[test]
    fn test_compute_moments() {
        let mut img = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

        // Draw a rectangle
        for row in 20..80 {
            for col in 30..70 {
                img.at_mut(row, col).unwrap()[0] = 255;
            }
        }

        let moments = compute_moments(&img).unwrap();

        assert!(moments.m00 > 0.0);
        assert!(moments.m10 > 0.0);
        assert!(moments.m01 > 0.0);
    }

    #[test]
    fn test_hu_moments() {
        let mut moments = Moments::new();
        moments.m00 = 1000.0;
        moments.nu20 = 0.1;
        moments.nu02 = 0.1;
        moments.nu11 = 0.05;

        let hu = hu_moments(&moments);

        assert_eq!(hu.len(), 7);
        assert!(hu[0] > 0.0); // hu1 should be positive for this example
    }

    #[test]
    fn test_contour_moments() {
        let contour = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let moments = contour_moments(&contour);

        assert!(moments.m00.abs() > 0.0);
    }

    #[test]
    fn test_centroid() {
        let mut moments = Moments::new();
        moments.m00 = 100.0;
        moments.m10 = 5000.0;
        moments.m01 = 3000.0;

        let (cx, cy) = moments.centroid();

        assert!((cx - 50.0).abs() < 1e-6);
        assert!((cy - 30.0).abs() < 1e-6);
    }
}
