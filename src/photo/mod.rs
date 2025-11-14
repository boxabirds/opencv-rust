pub mod hdr;
pub mod seam_carving;
pub mod super_resolution;
pub mod denoising;

pub use hdr::*;
pub use seam_carving::*;
pub use super_resolution::*;
pub use denoising::*;

use crate::core::Mat;
use crate::error::{Error, Result};

/// Inpaint image using the inpainting mask
pub fn inpaint(
    src: &Mat,
    inpaint_mask: &Mat,
    dst: &mut Mat,
    inpaint_radius: f64,
) -> Result<()> {
    if src.rows() != inpaint_mask.rows() || src.cols() != inpaint_mask.cols() {
        return Err(Error::InvalidDimensions(
            "Source and mask must have same dimensions".to_string(),
        ));
    }

    if inpaint_mask.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Inpaint mask must be single-channel".to_string(),
        ));
    }

    *dst = src.clone_mat();

    #[allow(clippy::cast_possible_truncation)]
    let radius = inpaint_radius as i32;

    // Simple inpainting: replace masked pixels with average of neighbors
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mask_pixel = inpaint_mask.at(row, col)?;

            if mask_pixel[0] > 0 {
                // Need to inpaint this pixel
                let mut sums = vec![0.0f64; src.channels()];
                let mut count = 0;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let y = row as i32 + dy;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let x = col as i32 + dx;

                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let rows_i32 = src.rows() as i32;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let cols_i32 = src.cols() as i32;

                        if y >= 0 && y < rows_i32 && x >= 0 && x < cols_i32 {
                            #[allow(clippy::cast_sign_loss)]
                            let y_usize = y as usize;
                            #[allow(clippy::cast_sign_loss)]
                            let x_usize = x as usize;

                            let neighbor_mask = inpaint_mask.at(y_usize, x_usize)?;

                            if neighbor_mask[0] == 0 {
                                // Not masked, use for inpainting
                                let neighbor_pixel = src.at(y_usize, x_usize)?;

                                #[allow(clippy::cast_possible_truncation)]
                                for ch in 0..src.channels() {
                                    sums[ch] += f64::from(neighbor_pixel[ch]);
                                }

                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    let dst_pixel = dst.at_mut(row, col)?;

                    #[allow(clippy::cast_possible_truncation)]
                    for ch in 0..src.channels() {
                        let clamped = (sums[ch] / f64::from(count)).clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let pixel_val = clamped as u8;
                        dst_pixel[ch] = pixel_val;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Denoise image using Non-local Means Denoising
pub fn fast_nl_means_denoising(
    src: &Mat,
    dst: &mut Mat,
    h: f32,
    template_window_size: i32,
    search_window_size: i32,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Non-local means denoising requires grayscale image".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, src.depth())?;

    let half_template = template_window_size / 2;
    let half_search = search_window_size / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut weight_sum = 0.0f32;
            let mut pixel_sum = 0.0f32;

            // Search in neighborhood
            for dy in -half_search..=half_search {
                for dx in -half_search..=half_search {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let y = row as i32 + dy;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let x = col as i32 + dx;

                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let rows_max = src.rows() as i32 - half_template;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let cols_max = src.cols() as i32 - half_template;

                    if y >= half_template && y < rows_max
                        && x >= half_template && x < cols_max
                    {
                        // Calculate similarity between patches
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let row_i32 = row as i32;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let col_i32 = col as i32;

                        let similarity = calculate_patch_distance(
                            src,
                            row_i32,
                            col_i32,
                            y,
                            x,
                            half_template,
                        )?;

                        let weight = libm::expf(-similarity / (h * h));

                        #[allow(clippy::cast_sign_loss)]
                        let y_usize = y as usize;
                        #[allow(clippy::cast_sign_loss)]
                        let x_usize = x as usize;
                        let neighbor_pixel = src.at(y_usize, x_usize)?;
                        pixel_sum += weight * f32::from(neighbor_pixel[0]);
                        weight_sum += weight;
                    }
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = if weight_sum > 0.0 {
                let clamped = (pixel_sum / weight_sum).clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pixel_val = clamped as u8;
                pixel_val
            } else {
                src.at(row, col)?[0]
            };
        }
    }

    Ok(())
}

fn calculate_patch_distance(
    img: &Mat,
    row1: i32,
    col1: i32,
    row2: i32,
    col2: i32,
    half_size: i32,
) -> Result<f32> {
    let mut dist = 0.0f32;
    let mut count = 0;

    for dy in -half_size..=half_size {
        for dx in -half_size..=half_size {
            let y1 = row1 + dy;
            let x1 = col1 + dx;
            let y2 = row2 + dy;
            let x2 = col2 + dx;

            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let rows_i32 = img.rows() as i32;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let cols_i32 = img.cols() as i32;

            if y1 >= 0 && y1 < rows_i32 && x1 >= 0 && x1 < cols_i32
                && y2 >= 0 && y2 < rows_i32 && x2 >= 0 && x2 < cols_i32
            {
                #[allow(clippy::cast_sign_loss)]
                let y1_usize = y1 as usize;
                #[allow(clippy::cast_sign_loss)]
                let x1_usize = x1 as usize;
                #[allow(clippy::cast_sign_loss)]
                let y2_usize = y2 as usize;
                #[allow(clippy::cast_sign_loss)]
                let x2_usize = x2 as usize;

                let p1 = img.at(y1_usize, x1_usize)?;
                let p2 = img.at(y2_usize, x2_usize)?;

                let diff = f32::from(p1[0]) - f32::from(p2[0]);
                dist += diff * diff;
                count += 1;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let count_f32 = count as f32;
    Ok(if count > 0 { dist / count_f32 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MatDepth, types::Scalar};

    #[test]
    fn test_inpaint() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mask = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(0.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        inpaint(&src, &mask, &mut dst, 3.0).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_fast_nl_means_denoising() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        fast_nl_means_denoising(&src, &mut dst, 10.0, 7, 21).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }
}
