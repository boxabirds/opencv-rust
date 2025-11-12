use crate::core::mat::Mat;
use crate::error::{Error, Result};

/// Type-safe accessors for Mat with different depths
impl Mat {
    /// Get f32 value at (row, col, channel)
    pub fn at_f32(&self, row: usize, col: usize, channel: usize) -> Result<f32> {
        if self.depth() != crate::core::MatDepth::F32 {
            return Err(Error::InvalidParameter(
                format!("Mat depth is {:?}, expected F32", self.depth())
            ));
        }

        if row >= self.rows() || col >= self.cols() || channel >= self.channels() {
            return Err(Error::OutOfRange(format!(
                "Index ({row}, {col}, {channel}) out of range"
            )));
        }

        let idx = (row * self.cols() + col) * self.channels() + channel;
        let byte_idx = idx * 4; // f32 is 4 bytes

        let bytes = &self.data()[byte_idx..byte_idx + 4];
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Ok(value)
    }

    /// Set f32 value at (row, col, channel)
    pub fn set_f32(&mut self, row: usize, col: usize, channel: usize, value: f32) -> Result<()> {
        if self.depth() != crate::core::MatDepth::F32 {
            return Err(Error::InvalidParameter(
                format!("Mat depth is {:?}, expected F32", self.depth())
            ));
        }

        if row >= self.rows() || col >= self.cols() || channel >= self.channels() {
            return Err(Error::OutOfRange(format!(
                "Index ({row}, {col}, {channel}) out of range"
            )));
        }

        let idx = (row * self.cols() + col) * self.channels() + channel;
        let byte_idx = idx * 4;

        let bytes = value.to_le_bytes();
        let data = self.data_mut();
        data[byte_idx..byte_idx + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Get f64 value at (row, col, channel)
    pub fn at_f64(&self, row: usize, col: usize, channel: usize) -> Result<f64> {
        if self.depth() != crate::core::MatDepth::F64 {
            return Err(Error::InvalidParameter(
                format!("Mat depth is {:?}, expected F64", self.depth())
            ));
        }

        if row >= self.rows() || col >= self.cols() || channel >= self.channels() {
            return Err(Error::OutOfRange(format!(
                "Index ({row}, {col}, {channel}) out of range"
            )));
        }

        let idx = (row * self.cols() + col) * self.channels() + channel;
        let byte_idx = idx * 8; // f64 is 8 bytes

        let bytes = &self.data()[byte_idx..byte_idx + 8];
        let value = f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        Ok(value)
    }

    /// Set f64 value at (row, col, channel)
    pub fn set_f64(&mut self, row: usize, col: usize, channel: usize, value: f64) -> Result<()> {
        if self.depth() != crate::core::MatDepth::F64 {
            return Err(Error::InvalidParameter(
                format!("Mat depth is {:?}, expected F64", self.depth())
            ));
        }

        if row >= self.rows() || col >= self.cols() || channel >= self.channels() {
            return Err(Error::OutOfRange(format!(
                "Index ({row}, {col}, {channel}) out of range"
            )));
        }

        let idx = (row * self.cols() + col) * self.channels() + channel;
        let byte_idx = idx * 8;

        let bytes = value.to_le_bytes();
        let data = self.data_mut();
        data[byte_idx..byte_idx + 8].copy_from_slice(&bytes);
        Ok(())
    }

    /// Get u16 value at (row, col, channel)
    pub fn at_u16(&self, row: usize, col: usize, channel: usize) -> Result<u16> {
        if self.depth() != crate::core::MatDepth::U16 {
            return Err(Error::InvalidParameter(
                format!("Mat depth is {:?}, expected U16", self.depth())
            ));
        }

        if row >= self.rows() || col >= self.cols() || channel >= self.channels() {
            return Err(Error::OutOfRange(format!(
                "Index ({row}, {col}, {channel}) out of range"
            )));
        }

        let idx = (row * self.cols() + col) * self.channels() + channel;
        let byte_idx = idx * 2;

        let bytes = &self.data()[byte_idx..byte_idx + 2];
        let value = u16::from_le_bytes([bytes[0], bytes[1]]);
        Ok(value)
    }

    /// Set u16 value at (row, col, channel)
    pub fn set_u16(&mut self, row: usize, col: usize, channel: usize, value: u16) -> Result<()> {
        if self.depth() != crate::core::MatDepth::U16 {
            return Err(Error::InvalidParameter(
                format!("Mat depth is {:?}, expected U16", self.depth())
            ));
        }

        if row >= self.rows() || col >= self.cols() || channel >= self.channels() {
            return Err(Error::OutOfRange(format!(
                "Index ({row}, {col}, {channel}) out of range"
            )));
        }

        let idx = (row * self.cols() + col) * self.channels() + channel;
        let byte_idx = idx * 2;

        let bytes = value.to_le_bytes();
        let data = self.data_mut();
        data[byte_idx..byte_idx + 2].copy_from_slice(&bytes);
        Ok(())
    }

    /// Convert Mat from one depth to another
    /// Normalizes between integer and floating-point types:
    /// - U8/U16 → F32/F64: divides by max value (255 or 65535)
    /// - F32/F64 → U8/U16: multiplies by max value (255 or 65535) and rounds
    pub fn convert_to(&self, target_depth: crate::core::MatDepth) -> Result<Mat> {
        if self.depth() == target_depth {
            return Ok(self.clone_mat());
        }

        use crate::core::MatDepth;
        let mut result = Mat::new(self.rows(), self.cols(), self.channels(), target_depth)?;

        for row in 0..self.rows() {
            for col in 0..self.cols() {
                for ch in 0..self.channels() {
                    // Read value and normalize to 0.0-1.0 range if integer type
                    let normalized_value = match self.depth() {
                        MatDepth::U8 => {
                            f64::from(self.at(row, col)?[ch]) / 255.0
                        }
                        MatDepth::U16 => {
                            f64::from(self.at_u16(row, col, ch)?) / 65535.0
                        }
                        MatDepth::F32 => {
                            f64::from(self.at_f32(row, col, ch)?)
                        }
                        MatDepth::F64 => {
                            self.at_f64(row, col, ch)?
                        }
                    };

                    // Write value, denormalizing from 0.0-1.0 if converting to integer type
                    match target_depth {
                        MatDepth::U8 => {
                            let pixel = result.at_mut(row, col)?;
                            // Multiply by 255 and round for F32/F64 → U8
                            let scaled = (normalized_value * 255.0).round();
                            pixel[ch] = scaled.clamp(0.0, 255.0) as u8;
                        }
                        MatDepth::U16 => {
                            // Multiply by 65535 and round for F32/F64 → U16
                            let scaled = (normalized_value * 65535.0).round();
                            result.set_u16(row, col, ch, scaled.clamp(0.0, 65535.0) as u16)?;
                        }
                        MatDepth::F32 => {
                            result.set_f32(row, col, ch, normalized_value as f32)?;
                        }
                        MatDepth::F64 => {
                            result.set_f64(row, col, ch, normalized_value)?;
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Mat, MatDepth, types::Scalar};

    #[test]
    fn test_f32_accessors() {
        let mut mat = Mat::new(10, 10, 1, MatDepth::F32).unwrap();

        mat.set_f32(5, 5, 0, 3.14159).unwrap();
        let val = mat.at_f32(5, 5, 0).unwrap();

        assert!((val - 3.14159).abs() < 1e-6);
    }

    #[test]
    fn test_f64_accessors() {
        let mut mat = Mat::new(10, 10, 1, MatDepth::F64).unwrap();

        mat.set_f64(3, 7, 0, 2.718281828).unwrap();
        let val = mat.at_f64(3, 7, 0).unwrap();

        assert!((val - 2.718281828).abs() < 1e-9);
    }

    #[test]
    fn test_u16_accessors() {
        let mut mat = Mat::new(5, 5, 1, MatDepth::U16).unwrap();

        mat.set_u16(2, 3, 0, 1234).unwrap();
        let val = mat.at_u16(2, 3, 0).unwrap();

        assert_eq!(val, 1234);
    }

    #[test]
    fn test_convert_u8_to_f32() {
        let mut mat_u8 = Mat::new(3, 3, 1, MatDepth::U8).unwrap();
        let pixel = mat_u8.at_mut(1, 1).unwrap();
        pixel[0] = 128;

        let mat_f32 = mat_u8.convert_to(MatDepth::F32).unwrap();
        let val = mat_f32.at_f32(1, 1, 0).unwrap();

        // Should normalize 128 → ~0.5
        assert!((val - 0.5019).abs() < 0.01, "128 should normalize to ~0.5");
    }

    #[test]
    fn test_convert_f32_to_u8() {
        let mut mat_f32 = Mat::new(3, 3, 1, MatDepth::F32).unwrap();
        mat_f32.set_f32(1, 1, 0, 0.7843).unwrap();  // 0.7843 * 255 ≈ 200

        let mat_u8 = mat_f32.convert_to(MatDepth::U8).unwrap();
        let val = mat_u8.at(1, 1).unwrap()[0];

        // Should denormalize ~0.7843 → 200
        assert!((val as i32 - 200).abs() <= 1, "~0.7843 should denormalize to ~200");
    }

    #[test]
    fn test_multichannel_f32() {
        let mut mat = Mat::new(2, 2, 3, MatDepth::F32).unwrap();

        mat.set_f32(0, 0, 0, 1.0).unwrap();
        mat.set_f32(0, 0, 1, 2.0).unwrap();
        mat.set_f32(0, 0, 2, 3.0).unwrap();

        assert_eq!(mat.at_f32(0, 0, 0).unwrap(), 1.0);
        assert_eq!(mat.at_f32(0, 0, 1).unwrap(), 2.0);
        assert_eq!(mat.at_f32(0, 0, 2).unwrap(), 3.0);
    }
}
