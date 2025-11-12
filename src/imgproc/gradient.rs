#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
use crate::core::Mat;
use crate::error::Result;

/// Compute gradient magnitude using Sobel derivatives
pub fn gradient_magnitude(src: &Mat, dst: &mut Mat) -> Result<()> {
    let rows = src.rows();
    let cols = src.cols();

    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
            let gx = (i32::from(src.at(y - 1, x + 1)?[0])
                - i32::from(src.at(y - 1, x - 1)?[0])
                + 2 * (i32::from(src.at(y, x + 1)?[0]) - i32::from(src.at(y, x - 1)?[0]))
                + i32::from(src.at(y + 1, x + 1)?[0])
                - i32::from(src.at(y + 1, x - 1)?[0]));

            // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
            let gy = (i32::from(src.at(y + 1, x - 1)?[0])
                + 2 * i32::from(src.at(y + 1, x)?[0])
                + i32::from(src.at(y + 1, x + 1)?[0])
                - i32::from(src.at(y - 1, x - 1)?[0])
                - 2 * i32::from(src.at(y - 1, x)?[0])
                - i32::from(src.at(y - 1, x + 1)?[0]));

            // Compute magnitude: sqrt(gx^2 + gy^2)
            let magnitude = ((gx * gx + gy * gy) as f64).sqrt();
            let clamped = magnitude.min(255.0).max(0.0);
            dst.at_mut(y, x)?[0] = clamped as u8;
        }
    }

    Ok(())
}
