use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

/// Multi-band blending for seamless image composition
pub struct MultiBandBlender {
    num_bands: usize,
    sharpness: f32,
}

impl MultiBandBlender {
    pub fn new(num_bands: usize) -> Self {
        Self {
            num_bands,
            sharpness: 0.02,
        }
    }

    pub fn with_sharpness(mut self, sharpness: f32) -> Self {
        self.sharpness = sharpness;
        self
    }

    pub fn blend(&self, images: &[Mat], masks: &[Mat]) -> Result<Mat> {
        if images.is_empty() || masks.is_empty() {
            return Err(Error::InvalidParameter("Need at least one image".to_string()));
        }

        if images.len() != masks.len() {
            return Err(Error::InvalidParameter(
                "Number of images must match number of masks".to_string(),
            ));
        }

        // Determine output size
        let rows = images.iter().map(|img| img.rows()).max().unwrap();
        let cols = images.iter().map(|img| img.cols()).max().unwrap();
        let channels = images[0].channels();

        // Build Laplacian pyramids for each image
        let mut pyramids = Vec::new();
        for img in images {
            let pyramid = self.build_laplacian_pyramid(img, self.num_bands)?;
            pyramids.push(pyramid);
        }

        // Build Gaussian pyramids for masks
        let mut mask_pyramids = Vec::new();
        for mask in masks {
            let pyramid = self.build_gaussian_pyramid(mask, self.num_bands)?;
            mask_pyramids.push(pyramid);
        }

        // Blend each band
        let mut blended_pyramid = Vec::new();

        for band in 0..self.num_bands {
            let band_rows = pyramids[0][band].rows();
            let band_cols = pyramids[0][band].cols();

            let mut blended_band = Mat::new(band_rows, band_cols, channels, MatDepth::F32)?;

            for row in 0..band_rows {
                for col in 0..band_cols {
                    for ch in 0..channels {
                        let mut weighted_sum = 0.0f32;
                        let mut weight_sum = 0.0f32;

                        for i in 0..images.len() {
                            if band < pyramids[i].len() && row < pyramids[i][band].rows()
                                && col < pyramids[i][band].cols()
                            {
                                let pixel_val = pyramids[i][band].at_f32(row, col, ch)?;
                                let weight = if row < mask_pyramids[i][band].rows()
                                    && col < mask_pyramids[i][band].cols()
                                {
                                    mask_pyramids[i][band].at_f32(row, col, 0)? / 255.0
                                } else {
                                    0.0
                                };

                                weighted_sum += pixel_val * weight;
                                weight_sum += weight;
                            }
                        }

                        let blended_val = if weight_sum > 0.0 {
                            weighted_sum / weight_sum
                        } else {
                            0.0
                        };

                        blended_band.set_f32(row, col, ch, blended_val)?;
                    }
                }
            }

            blended_pyramid.push(blended_band);
        }

        // Reconstruct from Laplacian pyramid
        let result = self.reconstruct_from_laplacian_pyramid(&blended_pyramid)?;

        // Convert back to U8
        let mut output = Mat::new(result.rows(), result.cols(), channels, MatDepth::U8)?;
        for row in 0..result.rows() {
            for col in 0..result.cols() {
                for ch in 0..channels {
                    let val = result.at_f32(row, col, ch)?.clamp(0.0, 255.0);
                    output.at_mut(row, col)?[ch] = val as u8;
                }
            }
        }

        Ok(output)
    }

    fn build_laplacian_pyramid(&self, image: &Mat, num_levels: usize) -> Result<Vec<Mat>> {
        // First build Gaussian pyramid
        let gaussian_pyramid = self.build_gaussian_pyramid(image, num_levels)?;

        let mut laplacian_pyramid = Vec::new();

        // Compute Laplacian = Gaussian[i] - upsample(Gaussian[i+1])
        for i in 0..gaussian_pyramid.len() - 1 {
            let current = &gaussian_pyramid[i];
            let next_upsampled = self.upsample(&gaussian_pyramid[i + 1], current.rows(), current.cols())?;

            let mut laplacian = Mat::new(current.rows(), current.cols(), current.channels(), MatDepth::F32)?;

            for row in 0..current.rows() {
                for col in 0..current.cols() {
                    for ch in 0..current.channels() {
                        let val1 = current.at_f32(row, col, ch)?;
                        let val2 = next_upsampled.at_f32(row, col, ch)?;
                        laplacian.set_f32(row, col, ch, val1 - val2)?;
                    }
                }
            }

            laplacian_pyramid.push(laplacian);
        }

        // Add the smallest Gaussian level
        laplacian_pyramid.push(gaussian_pyramid.last().unwrap().clone_mat());

        Ok(laplacian_pyramid)
    }

    fn build_gaussian_pyramid(&self, image: &Mat, num_levels: usize) -> Result<Vec<Mat>> {
        let mut pyramid = Vec::new();

        // Convert to F32
        let mut current = Mat::new(image.rows(), image.cols(), image.channels(), MatDepth::F32)?;
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                for ch in 0..image.channels() {
                    let val = image.at(row, col)?[ch] as f32;
                    current.set_f32(row, col, ch, val)?;
                }
            }
        }

        pyramid.push(current.clone_mat());

        // Build pyramid
        for _ in 1..num_levels {
            current = self.downsample(&current)?;
            pyramid.push(current.clone_mat());

            if current.rows() < 2 || current.cols() < 2 {
                break;
            }
        }

        Ok(pyramid)
    }

    fn downsample(&self, image: &Mat) -> Result<Mat> {
        let new_rows = (image.rows() / 2).max(1);
        let new_cols = (image.cols() / 2).max(1);

        let mut result = Mat::new(new_rows, new_cols, image.channels(), MatDepth::F32)?;

        for row in 0..new_rows {
            for col in 0..new_cols {
                for ch in 0..image.channels() {
                    let src_row = (row * 2).min(image.rows() - 1);
                    let src_col = (col * 2).min(image.cols() - 1);
                    let val = image.at_f32(src_row, src_col, ch)?;
                    result.set_f32(row, col, ch, val)?;
                }
            }
        }

        Ok(result)
    }

    fn upsample(&self, image: &Mat, target_rows: usize, target_cols: usize) -> Result<Mat> {
        let mut result = Mat::new(target_rows, target_cols, image.channels(), MatDepth::F32)?;

        let row_scale = image.rows() as f32 / target_rows as f32;
        let col_scale = image.cols() as f32 / target_cols as f32;

        for row in 0..target_rows {
            for col in 0..target_cols {
                let src_row = (row as f32 * row_scale).min(image.rows() as f32 - 1.0) as usize;
                let src_col = (col as f32 * col_scale).min(image.cols() as f32 - 1.0) as usize;

                for ch in 0..image.channels() {
                    let val = image.at_f32(src_row, src_col, ch)?;
                    result.set_f32(row, col, ch, val)?;
                }
            }
        }

        Ok(result)
    }

    fn reconstruct_from_laplacian_pyramid(&self, pyramid: &[Mat]) -> Result<Mat> {
        if pyramid.is_empty() {
            return Err(Error::InvalidParameter("Empty pyramid".to_string()));
        }

        // Start from smallest level
        let mut result = pyramid.last().unwrap().clone_mat();

        // Progressively add Laplacian levels
        for i in (0..pyramid.len() - 1).rev() {
            let laplacian = &pyramid[i];
            result = self.upsample(&result, laplacian.rows(), laplacian.cols())?;

            // Add Laplacian
            for row in 0..result.rows() {
                for col in 0..result.cols() {
                    for ch in 0..result.channels() {
                        let val1 = result.at_f32(row, col, ch)?;
                        let val2 = laplacian.at_f32(row, col, ch)?;
                        result.set_f32(row, col, ch, val1 + val2)?;
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Feather blending with distance transform
pub struct FeatherBlender {
    sharpness: f32,
}

impl FeatherBlender {
    pub fn new(sharpness: f32) -> Self {
        Self { sharpness }
    }

    pub fn blend(&self, images: &[Mat], masks: &[Mat]) -> Result<Mat> {
        if images.is_empty() || masks.is_empty() {
            return Err(Error::InvalidParameter("Need at least one image".to_string()));
        }

        let rows = images[0].rows();
        let cols = images[0].cols();
        let channels = images[0].channels();

        let mut result = Mat::new(rows, cols, channels, MatDepth::U8)?;

        // Compute weight maps with feathering
        let weight_maps = self.compute_weight_maps(masks)?;

        // Blend using weighted average
        for row in 0..rows {
            for col in 0..cols {
                for ch in 0..channels {
                    let mut weighted_sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for i in 0..images.len() {
                        if row < weight_maps[i].rows() && col < weight_maps[i].cols() {
                            let weight = weight_maps[i].at_f32(row, col, 0)?;
                            let pixel = images[i].at(row, col)?[ch] as f32;

                            weighted_sum += pixel * weight;
                            weight_sum += weight;
                        }
                    }

                    result.at_mut(row, col)?[ch] = if weight_sum > 0.0 {
                        (weighted_sum / weight_sum) as u8
                    } else {
                        0
                    };
                }
            }
        }

        Ok(result)
    }

    fn compute_weight_maps(&self, masks: &[Mat]) -> Result<Vec<Mat>> {
        let mut weight_maps = Vec::new();

        for mask in masks {
            let mut weights = Mat::new(mask.rows(), mask.cols(), 1, MatDepth::F32)?;

            // Compute distance from mask boundary
            for row in 0..mask.rows() {
                for col in 0..mask.cols() {
                    if mask.at(row, col)?[0] > 0 {
                        let dist = self.distance_to_boundary(mask, row, col)?;
                        let weight = 1.0 / (1.0 + (-self.sharpness * dist).exp());
                        weights.set_f32(row, col, 0, weight)?;
                    } else {
                        weights.set_f32(row, col, 0, 0.0)?;
                    }
                }
            }

            weight_maps.push(weights);
        }

        Ok(weight_maps)
    }

    fn distance_to_boundary(&self, mask: &Mat, row: usize, col: usize) -> Result<f32> {
        let mut min_dist = f32::MAX;

        // Simple boundary distance (check 8-connected neighbors)
        let search_radius = 5;

        for dy in -(search_radius as i32)..=(search_radius as i32) {
            for dx in -(search_radius as i32)..=(search_radius as i32) {
                let y = (row as i32 + dy).max(0).min(mask.rows() as i32 - 1) as usize;
                let x = (col as i32 + dx).max(0).min(mask.cols() as i32 - 1) as usize;

                if mask.at(y, x)?[0] == 0 {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    min_dist = min_dist.min(dist);
                }
            }
        }

        Ok(min_dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_multiband_blender() {
        let img1 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let img2 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(150.0)).unwrap();

        let mask1 = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();
        let mask2 = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();

        let images = vec![img1, img2];
        let masks = vec![mask1, mask2];

        let blender = MultiBandBlender::new(3);
        let result = blender.blend(&images, &masks).unwrap();

        assert_eq!(result.rows(), 100);
        assert_eq!(result.cols(), 100);
    }

    #[test]
    fn test_feather_blender() {
        let img1 = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let img2 = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(150.0)).unwrap();

        let mask1 = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();
        let mask2 = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();

        let images = vec![img1, img2];
        let masks = vec![mask1, mask2];

        let blender = FeatherBlender::new(0.1);
        let result = blender.blend(&images, &masks).unwrap();

        assert_eq!(result.rows(), 50);
    }
}
