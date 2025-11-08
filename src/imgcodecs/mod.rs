use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, Luma};
use std::path::Path;

/// Read an image from file
pub fn imread<P: AsRef<Path>>(path: P) -> Result<Mat> {
    let img = image::open(path)?;

    match img {
        DynamicImage::ImageRgb8(buffer) => {
            let (width, height) = buffer.dimensions();
            let data = buffer.into_raw();
            Mat::from_raw(data, height as usize, width as usize, 3, MatDepth::U8)
        }
        DynamicImage::ImageRgba8(buffer) => {
            let (width, height) = buffer.dimensions();
            let data = buffer.into_raw();
            Mat::from_raw(data, height as usize, width as usize, 4, MatDepth::U8)
        }
        DynamicImage::ImageLuma8(buffer) => {
            let (width, height) = buffer.dimensions();
            let data = buffer.into_raw();
            Mat::from_raw(data, height as usize, width as usize, 1, MatDepth::U8)
        }
        _ => {
            // Convert any other format to RGB8
            let rgb_img = img.to_rgb8();
            let (width, height) = rgb_img.dimensions();
            let data = rgb_img.into_raw();
            Mat::from_raw(data, height as usize, width as usize, 3, MatDepth::U8)
        }
    }
}

/// Write an image to file
pub fn imwrite<P: AsRef<Path>>(path: P, mat: &Mat) -> Result<()> {
    if mat.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "imwrite only supports U8 depth".to_string(),
        ));
    }

    match mat.channels() {
        1 => {
            let buffer = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                mat.cols() as u32,
                mat.rows() as u32,
                mat.data().to_vec(),
            )
            .ok_or_else(|| Error::InvalidDimensions("Failed to create image buffer".to_string()))?;

            buffer.save(path)?;
        }
        3 => {
            let buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                mat.cols() as u32,
                mat.rows() as u32,
                mat.data().to_vec(),
            )
            .ok_or_else(|| Error::InvalidDimensions("Failed to create image buffer".to_string()))?;

            buffer.save(path)?;
        }
        4 => {
            let buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
                mat.cols() as u32,
                mat.rows() as u32,
                mat.data().to_vec(),
            )
            .ok_or_else(|| Error::InvalidDimensions("Failed to create image buffer".to_string()))?;

            buffer.save(path)?;
        }
        _ => {
            return Err(Error::UnsupportedOperation(format!(
                "imwrite doesn't support {} channels",
                mat.channels()
            )));
        }
    }

    Ok(())
}

/// Read flags for imread
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImreadFlag {
    Color,
    Grayscale,
    Unchanged,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_write_and_read() {
        let mat = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::from_rgb(255, 0, 0))
            .unwrap();

        let temp_path = "/tmp/test_opencv_rust.png";
        imwrite(temp_path, &mat).unwrap();

        let loaded = imread(temp_path).unwrap();
        assert_eq!(loaded.rows(), mat.rows());
        assert_eq!(loaded.cols(), mat.cols());
    }
}
