use crate::error::{Error, Result};

/// Multi-dimensional blob for neural network data
/// Shape convention: [batch, channels, height, width] (NCHW)
#[derive(Clone, Debug)]
pub struct Blob {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Blob {
    /// Create new blob with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Create blob with data
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(Error::InvalidDimensions(
                format!("Data size {} doesn't match shape {:?}", data.len(), shape)
            ));
        }

        Ok(Self { data, shape })
    }

    /// Create blob from image (HWC to NCHW)
    pub fn from_image(image: &crate::core::Mat) -> Result<Self> {
        let height = image.rows();
        let width = image.cols();
        let channels = image.channels();

        let mut data = Vec::with_capacity(channels * height * width);

        // Convert HWC to CHW
        for c in 0..channels {
            for row in 0..height {
                for col in 0..width {
                    let pixel = image.at(row, col)?;
                    data.push(pixel[c] as f32 / 255.0);
                }
            }
        }

        Ok(Self {
            data,
            shape: vec![1, channels, height, width],
        })
    }

    /// Get blob shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get total number of elements
    pub fn total(&self) -> usize {
        self.data.len()
    }

    /// Get data reference
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable data reference
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Reshape blob
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<()> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(Error::InvalidDimensions(
                format!("Cannot reshape {} elements to {:?}", self.data.len(), new_shape)
            ));
        }

        self.shape = new_shape;
        Ok(())
    }

    /// Get value at index
    pub fn at(&self, indices: &[usize]) -> Result<f32> {
        let idx = self.compute_index(indices)?;
        Ok(self.data[idx])
    }

    /// Set value at index
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        let idx = self.compute_index(indices)?;
        self.data[idx] = value;
        Ok(())
    }

    fn compute_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(Error::InvalidDimensions(
                format!("Expected {} indices, got {}", self.shape.len(), indices.len())
            ));
        }

        let mut idx = 0;
        let mut stride = 1;

        for i in (0..indices.len()).rev() {
            if indices[i] >= self.shape[i] {
                return Err(Error::InvalidParameter(
                    format!("Index {} out of bounds for dimension {}", indices[i], i)
                ));
            }
            idx += indices[i] * stride;
            stride *= self.shape[i];
        }

        Ok(idx)
    }

    /// Clone blob data
    pub fn clone_blob(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
}

/// Blob creation from Mat with preprocessing
pub fn blob_from_image(
    image: &crate::core::Mat,
    scale_factor: f32,
    mean: &[f32; 3],
    swap_rb: bool,
) -> Result<Blob> {
    let height = image.rows();
    let width = image.cols();
    let channels = image.channels().min(3);

    let mut data = Vec::with_capacity(channels * height * width);

    // Convert HWC to CHW with preprocessing
    for c in 0..channels {
        for row in 0..height {
            for col in 0..width {
                let pixel = image.at(row, col)?;

                // Optionally swap R and B channels
                let channel_idx = if swap_rb && channels == 3 {
                    2 - c
                } else {
                    c
                };

                let value = (pixel[channel_idx] as f32 * scale_factor) - mean[c];
                data.push(value);
            }
        }
    }

    Ok(Blob {
        data,
        shape: vec![1, channels, height, width],
    })
}

/// Blob creation from multiple images (batch)
pub fn blob_from_images(
    images: &[&crate::core::Mat],
    scale_factor: f32,
    mean: &[f32; 3],
    swap_rb: bool,
) -> Result<Blob> {
    if images.is_empty() {
        return Err(Error::InvalidParameter("No images provided".to_string()));
    }

    let batch_size = images.len();
    let height = images[0].rows();
    let width = images[0].cols();
    let channels = images[0].channels().min(3);

    let mut data = Vec::with_capacity(batch_size * channels * height * width);

    for image in images {
        if image.rows() != height || image.cols() != width {
            return Err(Error::InvalidDimensions(
                "All images must have the same size".to_string()
            ));
        }

        // Convert HWC to CHW with preprocessing
        for c in 0..channels {
            for row in 0..height {
                for col in 0..width {
                    let pixel = image.at(row, col)?;

                    let channel_idx = if swap_rb && channels == 3 {
                        2 - c
                    } else {
                        c
                    };

                    let value = (pixel[channel_idx] as f32 * scale_factor) - mean[c];
                    data.push(value);
                }
            }
        }
    }

    Ok(Blob {
        data,
        shape: vec![batch_size, channels, height, width],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Mat, MatDepth, types::Scalar};

    #[test]
    fn test_blob_creation() {
        let blob = Blob::new(vec![1, 3, 224, 224]);
        assert_eq!(blob.shape(), &[1, 3, 224, 224]);
        assert_eq!(blob.total(), 1 * 3 * 224 * 224);
    }

    #[test]
    fn test_blob_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let blob = Blob::from_data(data, vec![2, 2]).unwrap();
        assert_eq!(blob.shape(), &[2, 2]);
        assert_eq!(blob.at(&[0, 0]).unwrap(), 1.0);
        assert_eq!(blob.at(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_blob_from_image() {
        let img = Mat::new_with_default(64, 64, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let blob = Blob::from_image(&img).unwrap();

        assert_eq!(blob.shape(), &[1, 3, 64, 64]);
        // Value should be ~0.5 (128/255)
        assert!((blob.at(&[0, 0, 0, 0]).unwrap() - 0.502).abs() < 0.01);
    }

    #[test]
    fn test_blob_reshape() {
        let mut blob = Blob::new(vec![2, 3, 4]);
        blob.reshape(vec![6, 4]).unwrap();
        assert_eq!(blob.shape(), &[6, 4]);
    }

    #[test]
    fn test_blob_preprocessing() {
        let img = Mat::new_with_default(32, 32, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mean = [0.5, 0.5, 0.5];
        let blob = blob_from_image(&img, 1.0 / 255.0, &mean, false).unwrap();

        assert_eq!(blob.shape(), &[1, 3, 32, 32]);
        // (128/255) - 0.5 ≈ 0.0
        assert!(blob.at(&[0, 0, 0, 0]).unwrap().abs() < 0.01);
    }
}
