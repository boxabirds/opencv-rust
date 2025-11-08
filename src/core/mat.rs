use crate::error::{Error, Result};
use crate::core::types::{Size, Rect, Scalar};
use ndarray::Array3;

/// Matrix type representing an image or general n-dimensional data
#[derive(Debug, Clone)]
pub struct Mat {
    data: Vec<u8>,
    rows: usize,
    cols: usize,
    channels: usize,
    depth: MatDepth,
}

/// Matrix depth (element type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatDepth {
    U8,
    U16,
    F32,
    F64,
}

impl MatDepth {
    pub fn size(&self) -> usize {
        match self {
            MatDepth::U8 => 1,
            MatDepth::U16 => 2,
            MatDepth::F32 => 4,
            MatDepth::F64 => 8,
        }
    }
}

impl Mat {
    /// Create a new Mat with given dimensions and channels
    pub fn new(rows: usize, cols: usize, channels: usize, depth: MatDepth) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::InvalidDimensions(
                "Rows and columns must be greater than 0".to_string(),
            ));
        }

        let total_size = rows * cols * channels * depth.size();
        let data = vec![0u8; total_size];

        Ok(Self {
            data,
            rows,
            cols,
            channels,
            depth,
        })
    }

    /// Create a Mat filled with a scalar value
    pub fn new_with_default(
        rows: usize,
        cols: usize,
        channels: usize,
        depth: MatDepth,
        value: Scalar,
    ) -> Result<Self> {
        let mut mat = Self::new(rows, cols, channels, depth)?;
        mat.set_to(value)?;
        Ok(mat)
    }

    /// Create a Mat from raw data
    pub fn from_raw(
        data: Vec<u8>,
        rows: usize,
        cols: usize,
        channels: usize,
        depth: MatDepth,
    ) -> Result<Self> {
        let expected_size = rows * cols * channels * depth.size();
        if data.len() != expected_size {
            return Err(Error::InvalidDimensions(format!(
                "Data size {} doesn't match expected size {}",
                data.len(),
                expected_size
            )));
        }

        Ok(Self {
            data,
            rows,
            cols,
            channels,
            depth,
        })
    }

    /// Get dimensions
    pub fn size(&self) -> Size {
        Size::new(self.cols as i32, self.rows as i32)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn depth(&self) -> MatDepth {
        self.depth
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Get raw data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get pixel value at (row, col)
    pub fn at(&self, row: usize, col: usize) -> Result<&[u8]> {
        if row >= self.rows || col >= self.cols {
            return Err(Error::OutOfRange(format!(
                "Index ({}, {}) out of range for {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

        let idx = (row * self.cols + col) * self.channels * self.depth.size();
        let end = idx + self.channels * self.depth.size();
        Ok(&self.data[idx..end])
    }

    /// Get mutable pixel value at (row, col)
    pub fn at_mut(&mut self, row: usize, col: usize) -> Result<&mut [u8]> {
        if row >= self.rows || col >= self.cols {
            return Err(Error::OutOfRange(format!(
                "Index ({}, {}) out of range for {}x{} matrix",
                row, col, self.rows, self.cols
            )));
        }

        let idx = (row * self.cols + col) * self.channels * self.depth.size();
        let end = idx + self.channels * self.depth.size();
        Ok(&mut self.data[idx..end])
    }

    /// Set all pixels to a scalar value
    pub fn set_to(&mut self, value: Scalar) -> Result<()> {
        if self.depth != MatDepth::U8 {
            return Err(Error::UnsupportedOperation(
                "set_to only supports U8 depth".to_string(),
            ));
        }

        let num_channels = self.channels;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let pixel = self.at_mut(row, col)?;
                for ch in 0..num_channels.min(4) {
                    pixel[ch] = value.val[ch] as u8;
                }
            }
        }
        Ok(())
    }

    /// Clone a region of interest (ROI)
    pub fn roi(&self, rect: Rect) -> Result<Mat> {
        if rect.x < 0 || rect.y < 0 {
            return Err(Error::OutOfRange("ROI coordinates must be non-negative".to_string()));
        }

        let x = rect.x as usize;
        let y = rect.y as usize;
        let w = rect.width as usize;
        let h = rect.height as usize;

        if x + w > self.cols || y + h > self.rows {
            return Err(Error::OutOfRange("ROI exceeds matrix dimensions".to_string()));
        }

        let mut result = Mat::new(h, w, self.channels, self.depth)?;

        for row in 0..h {
            for col in 0..w {
                let src_pixel = self.at(y + row, x + col)?;
                let dst_pixel = result.at_mut(row, col)?;
                dst_pixel.copy_from_slice(src_pixel);
            }
        }

        Ok(result)
    }

    /// Clone the matrix
    pub fn clone_mat(&self) -> Mat {
        Self {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
            channels: self.channels,
            depth: self.depth,
        }
    }

    /// Convert to ndarray for easier manipulation
    pub fn to_array3(&self) -> Result<Array3<u8>> {
        if self.depth != MatDepth::U8 {
            return Err(Error::UnsupportedOperation(
                "to_array3 only supports U8 depth".to_string(),
            ));
        }

        let shape = (self.rows, self.cols, self.channels);
        Array3::from_shape_vec(shape, self.data.clone())
            .map_err(|e| Error::InvalidDimensions(e.to_string()))
    }

    /// Create Mat from ndarray
    pub fn from_array3(arr: Array3<u8>) -> Result<Self> {
        let shape = arr.shape();
        let rows = shape[0];
        let cols = shape[1];
        let channels = shape[2];

        let data = arr.into_raw_vec();
        Self::from_raw(data, rows, cols, channels, MatDepth::U8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_creation() {
        let mat = Mat::new(100, 100, 3, MatDepth::U8).unwrap();
        assert_eq!(mat.rows(), 100);
        assert_eq!(mat.cols(), 100);
        assert_eq!(mat.channels(), 3);
    }

    #[test]
    fn test_mat_at() {
        let mut mat = Mat::new(10, 10, 3, MatDepth::U8).unwrap();
        let pixel = mat.at_mut(5, 5).unwrap();
        pixel[0] = 255;
        pixel[1] = 128;
        pixel[2] = 64;

        let pixel = mat.at(5, 5).unwrap();
        assert_eq!(pixel[0], 255);
        assert_eq!(pixel[1], 128);
        assert_eq!(pixel[2], 64);
    }

    #[test]
    fn test_mat_roi() {
        let mat = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(255.0)).unwrap();
        let roi = mat.roi(Rect::new(10, 10, 20, 20)).unwrap();
        assert_eq!(roi.rows(), 20);
        assert_eq!(roi.cols(), 20);
    }
}
