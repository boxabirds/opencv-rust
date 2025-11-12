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
    #[must_use] 
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
    ///
    /// This is the opencv-rust compatible constructor
    pub fn new_rows_cols(rows: usize, cols: usize, channels: usize, depth: MatDepth) -> Result<Self> {
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

    /// Create a new Mat with given dimensions and channels
    ///
    /// Convenience alias for `new_rows_cols`
    pub fn new(rows: usize, cols: usize, channels: usize, depth: MatDepth) -> Result<Self> {
        Self::new_rows_cols(rows, cols, channels, depth)
    }

    /// Create a new Mat from Size
    pub fn new_size(size: Size, channels: usize, depth: MatDepth) -> Result<Self> {
        Self::new_rows_cols(size.height as usize, size.width as usize, channels, depth)
    }

    /// Create a Mat filled with a scalar value (opencv-rust compatible name)
    pub fn new_rows_cols_with_default(
        rows: usize,
        cols: usize,
        channels: usize,
        depth: MatDepth,
        value: Scalar,
    ) -> Result<Self> {
        let mut mat = Self::new_rows_cols(rows, cols, channels, depth)?;
        mat.set_to(value)?;
        Ok(mat)
    }

    /// Create a Mat filled with a scalar value (convenience alias)
    pub fn new_with_default(
        rows: usize,
        cols: usize,
        channels: usize,
        depth: MatDepth,
        value: Scalar,
    ) -> Result<Self> {
        Self::new_rows_cols_with_default(rows, cols, channels, depth, value)
    }

    /// Create a Mat filled with a scalar value from Size
    pub fn new_size_with_default(
        size: Size,
        channels: usize,
        depth: MatDepth,
        value: Scalar,
    ) -> Result<Self> {
        Self::new_rows_cols_with_default(
            size.height as usize,
            size.width as usize,
            channels,
            depth,
            value,
        )
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

    /// Create a Mat from a slice (copies the data)
    pub fn from_slice(
        slice: &[u8],
        rows: usize,
        cols: usize,
        channels: usize,
        depth: MatDepth,
    ) -> Result<Self> {
        Self::from_raw(slice.to_vec(), rows, cols, channels, depth)
    }

    /// Create a Mat from a byte slice
    pub fn from_bytes(
        bytes: &[u8],
        rows: usize,
        cols: usize,
        channels: usize,
        depth: MatDepth,
    ) -> Result<Self> {
        Self::from_slice(bytes, rows, cols, channels, depth)
    }

    /// Create a zero-filled matrix
    pub fn zeros(rows: usize, cols: usize, channels: usize, depth: MatDepth) -> Result<Self> {
        Self::new_rows_cols(rows, cols, channels, depth)
    }

    /// Create a zero-filled matrix from Size
    pub fn zeros_size(size: Size, channels: usize, depth: MatDepth) -> Result<Self> {
        Self::new_size(size, channels, depth)
    }

    /// Create a one-filled matrix
    pub fn ones(rows: usize, cols: usize, channels: usize, depth: MatDepth) -> Result<Self> {
        Self::new_rows_cols_with_default(rows, cols, channels, depth, Scalar::all(1.0))
    }

    /// Create a one-filled matrix from Size
    pub fn ones_size(size: Size, channels: usize, depth: MatDepth) -> Result<Self> {
        Self::new_size_with_default(size, channels, depth, Scalar::all(1.0))
    }

    /// Create an identity matrix
    pub fn eye(rows: usize, cols: usize, depth: MatDepth) -> Result<Self> {
        let mut mat = Self::zeros(rows, cols, 1, depth)?;

        let min_dim = rows.min(cols);
        for i in 0..min_dim {
            match depth {
                MatDepth::U8 => {
                    let pixel = mat.at_mut(i, i)?;
                    pixel[0] = 1;
                }
                MatDepth::U16 => {
                    let pixel = mat.at_mut(i, i)?;
                    pixel[0] = 1;
                    pixel[1] = 0;
                }
                MatDepth::F32 => {
                    let pixel = mat.at_mut(i, i)?;
                    let bytes = 1.0f32.to_ne_bytes();
                    pixel[..4].copy_from_slice(&bytes);
                }
                MatDepth::F64 => {
                    let pixel = mat.at_mut(i, i)?;
                    let bytes = 1.0f64.to_ne_bytes();
                    pixel[..8].copy_from_slice(&bytes);
                }
            }
        }

        Ok(mat)
    }

    /// Create an identity matrix from Size
    pub fn eye_size(size: Size, depth: MatDepth) -> Result<Self> {
        Self::eye(size.height as usize, size.width as usize, depth)
    }

    /// Get dimensions
    #[must_use] 
    pub fn size(&self) -> Size {
        Size::new(self.cols as i32, self.rows as i32)
    }

    #[must_use] 
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use] 
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get width (same as cols, opencv-rust compatible)
    #[must_use] 
    pub fn width(&self) -> usize {
        self.cols
    }

    /// Get height (same as rows, opencv-rust compatible)
    #[must_use] 
    pub fn height(&self) -> usize {
        self.rows
    }

    #[must_use] 
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[must_use] 
    pub fn depth(&self) -> MatDepth {
        self.depth
    }

    #[must_use] 
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Get element size in bytes
    #[must_use] 
    pub fn elem_size(&self) -> usize {
        self.channels * self.depth.size()
    }

    /// Get element size for a single channel in bytes
    #[must_use] 
    pub fn elem_size1(&self) -> usize {
        self.depth.size()
    }

    /// Get total number of elements
    #[must_use] 
    pub fn total(&self) -> usize {
        self.rows * self.cols
    }

    /// Get the type identifier (combining depth and channels)
    /// Returns a value compatible with `OpenCV`'s type system
    #[must_use] 
    pub fn type_(&self) -> i32 {
        // OpenCV type encoding: ((depth) + ((channels-1) << 3))
        let depth_val = match self.depth {
            MatDepth::U8 => 0,
            MatDepth::U16 => 2,
            MatDepth::F32 => 5,
            MatDepth::F64 => 6,
        };
        depth_val + ((self.channels - 1) << 3) as i32
    }

    /// Get number of dimensions (always 2 for this implementation)
    #[must_use] 
    pub fn dims(&self) -> i32 {
        2
    }

    /// Check if matrix data is stored continuously
    /// Returns true if there are no gaps between rows
    #[must_use] 
    pub fn is_continuous(&self) -> bool {
        // In our implementation, data is always stored continuously
        true
    }

    /// Get the number of bytes each row occupies
    #[must_use] 
    pub fn step1(&self) -> usize {
        self.cols * self.elem_size()
    }

    /// Get raw data
    #[must_use] 
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get pixel value at (row, col)
    /// Panics if index is out of bounds
    pub fn at(&self, row: usize, col: usize) -> Result<&[u8]> {
        assert!(
            row < self.rows && col < self.cols,
            "Index ({}, {}) out of range for {}x{} matrix",
            row, col, self.rows, self.cols
        );

        let idx = (row * self.cols + col) * self.channels * self.depth.size();
        let end = idx + self.channels * self.depth.size();
        Ok(&self.data[idx..end])
    }

    /// Get mutable pixel value at (row, col)
    /// Panics if index is out of bounds
    pub fn at_mut(&mut self, row: usize, col: usize) -> Result<&mut [u8]> {
        assert!(
            row < self.rows && col < self.cols,
            "Index ({}, {}) out of range for {}x{} matrix",
            row, col, self.rows, self.cols
        );

        let idx = (row * self.cols + col) * self.channels * self.depth.size();
        let end = idx + self.channels * self.depth.size();
        Ok(&mut self.data[idx..end])
    }

    /// Get pixel value at (row, col) without bounds checking
    ///
    /// # Safety
    ///
    /// Caller must ensure that row < rows and col < cols
    #[inline(always)]
    #[must_use] 
    pub unsafe fn at_unchecked(&self, row: usize, col: usize) -> &[u8] {
        let idx = (row * self.cols + col) * self.channels * self.depth.size();
        let end = idx + self.channels * self.depth.size();
        self.data.get_unchecked(idx..end)
    }

    /// Get mutable pixel value at (row, col) without bounds checking
    ///
    /// # Safety
    ///
    /// Caller must ensure that row < rows and col < cols
    #[inline(always)]
    pub unsafe fn at_mut_unchecked(&mut self, row: usize, col: usize) -> &mut [u8] {
        let idx = (row * self.cols + col) * self.channels * self.depth.size();
        let end = idx + self.channels * self.depth.size();
        self.data.get_unchecked_mut(idx..end)
    }

    /// Set all pixels to a scalar value
    pub fn set_to(&mut self, value: Scalar) -> Result<()> {
        if self.depth != MatDepth::U8 {
            return Err(Error::UnsupportedOperation(
                "set_to only supports U8 depth".to_string(),
            ));
        }

        let num_channels = self.channels.min(4);
        for row in 0..self.rows {
            for col in 0..self.cols {
                let pixel = self.at_mut(row, col)?;
                for (ch, &val) in value.val.iter().take(num_channels).enumerate() {
                    pixel[ch] = val as u8;
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

        let mut result = Mat::new_rows_cols(h, w, self.channels, self.depth)?;

        for row in 0..h {
            for col in 0..w {
                let src_pixel = self.at(y + row, x + col)?;
                let dst_pixel = result.at_mut(row, col)?;
                dst_pixel.copy_from_slice(src_pixel);
            }
        }

        Ok(result)
    }

    /// Create a mutable region of interest (ROI)
    /// Note: This returns a new Mat, not a view, as we don't support shared mutable views
    pub fn roi_mut(&mut self, rect: Rect) -> Result<Mat> {
        self.roi(rect)
    }

    /// Extract a row range
    pub fn rowscols(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Result<Mat> {
        if row_end > self.rows || col_end > self.cols {
            return Err(Error::OutOfRange("Row/column range exceeds matrix dimensions".to_string()));
        }

        let rect = Rect::new(
            col_start as i32,
            row_start as i32,
            (col_end - col_start) as i32,
            (row_end - row_start) as i32,
        );
        self.roi(rect)
    }

    /// Extract a mutable row range
    pub fn rowscols_mut(&mut self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Result<Mat> {
        self.rowscols(row_start, row_end, col_start, col_end)
    }

    /// Copy this matrix to another matrix
    pub fn copy_to(&self, dst: &mut Mat) -> Result<()> {
        if self.rows != dst.rows || self.cols != dst.cols || self.channels != dst.channels || self.depth != dst.depth {
            return Err(Error::InvalidDimensions(
                "Source and destination matrices must have the same dimensions".to_string(),
            ));
        }

        dst.data.copy_from_slice(&self.data);
        Ok(())
    }

    /// Clone the matrix
    #[must_use] 
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
