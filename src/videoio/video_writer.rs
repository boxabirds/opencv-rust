use crate::core::Mat;
use crate::error::{Error, Result};
use std::path::Path;

/// Video writer
pub struct VideoWriter {
    path: String,
    fourcc: FourCC,
    fps: f64,
    frame_width: usize,
    frame_height: usize,
    is_color: bool,
    is_opened: bool,
    frames: Vec<Mat>,
}

impl VideoWriter {
    /// Create new video writer
    pub fn new<P: AsRef<Path>>(
        path: P,
        fourcc: FourCC,
        fps: f64,
        frame_width: usize,
        frame_height: usize,
        is_color: bool,
    ) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        Ok(Self {
            path: path_str,
            fourcc,
            fps,
            frame_width,
            frame_height,
            is_color,
            is_opened: true,
            frames: Vec::new(),
        })
    }

    /// Check if video writer is opened
    #[must_use] 
    pub fn is_opened(&self) -> bool {
        self.is_opened
    }

    /// Write frame to video
    pub fn write(&mut self, frame: &Mat) -> Result<()> {
        if !self.is_opened {
            return Err(Error::InvalidParameter("Video writer not opened".to_string()));
        }

        if frame.rows() != self.frame_height || frame.cols() != self.frame_width {
            return Err(Error::InvalidDimensions(format!(
                "Frame size {}x{} doesn't match writer size {}x{}",
                frame.cols(),
                frame.rows(),
                self.frame_width,
                self.frame_height
            )));
        }

        if self.is_color && frame.channels() != 3 {
            return Err(Error::InvalidParameter(
                "Color video requires 3-channel frames".to_string(),
            ));
        }

        if !self.is_color && frame.channels() != 1 {
            return Err(Error::InvalidParameter(
                "Grayscale video requires single-channel frames".to_string(),
            ));
        }

        // Store frame (in real implementation, would encode and write to file)
        self.frames.push(frame.clone_mat());

        Ok(())
    }

    /// Release video writer
    pub fn release(&mut self) -> Result<()> {
        if !self.is_opened {
            return Ok(());
        }

        // In real implementation, finalize video file
        // For now, just mark as closed
        self.is_opened = false;

        Ok(())
    }

    /// Get number of frames written
    #[must_use] 
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Get video properties
    #[must_use] 
    pub fn get_fps(&self) -> f64 {
        self.fps
    }

    #[must_use] 
    pub fn get_frame_size(&self) -> (usize, usize) {
        (self.frame_width, self.frame_height)
    }

    #[must_use] 
    pub fn get_fourcc(&self) -> FourCC {
        self.fourcc
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

/// Four-character code for video codecs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourCC {
    /// MJPEG codec
    MJPEG,
    /// H.264 codec
    H264,
    /// H.265/HEVC codec
    H265,
    /// VP8 codec
    VP8,
    /// VP9 codec
    VP9,
    /// MPEG-4 Part 2 codec
    MP4V,
    /// Xvid MPEG-4 codec
    XVID,
    /// Uncompressed YUV 4:2:0
    I420,
    /// Motion JPEG 2000
    MJ2C,
    /// Custom four-character code
    Custom([u8; 4]),
}

impl FourCC {
    /// Create `FourCC` from four characters
    #[must_use] 
    pub fn from_chars(c1: u8, c2: u8, c3: u8, c4: u8) -> Self {
        FourCC::Custom([c1, c2, c3, c4])
    }

    /// Create `FourCC` from string
    pub fn from_str(s: &str) -> Result<Self> {
        if s.len() != 4 {
            return Err(Error::InvalidParameter(
                "FourCC must be exactly 4 characters".to_string(),
            ));
        }

        let bytes = s.as_bytes();
        Ok(FourCC::Custom([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Convert `FourCC` to integer
    #[must_use] 
    pub fn to_int(&self) -> i32 {
        let bytes = match self {
            FourCC::MJPEG => [b'M', b'J', b'P', b'G'],
            FourCC::H264 => [b'H', b'2', b'6', b'4'],
            FourCC::H265 => [b'H', b'2', b'6', b'5'],
            FourCC::VP8 => [b'V', b'P', b'8', b'0'],
            FourCC::VP9 => [b'V', b'P', b'9', b'0'],
            FourCC::MP4V => [b'M', b'P', b'4', b'V'],
            FourCC::XVID => [b'X', b'V', b'I', b'D'],
            FourCC::I420 => [b'I', b'4', b'2', b'0'],
            FourCC::MJ2C => [b'M', b'J', b'2', b'C'],
            FourCC::Custom(bytes) => *bytes,
        };

        i32::from_le_bytes(bytes)
    }

    /// Create `FourCC` from integer
    #[must_use] 
    pub fn from_int(code: i32) -> Self {
        let bytes = code.to_le_bytes();
        FourCC::Custom(bytes)
    }
}

/// Get available video codecs
#[must_use] 
pub fn get_available_codecs() -> Vec<FourCC> {
    vec![
        FourCC::MJPEG,
        FourCC::H264,
        FourCC::H265,
        FourCC::VP8,
        FourCC::VP9,
        FourCC::MP4V,
        FourCC::XVID,
        FourCC::I420,
    ]
}

/// Check if codec is available
#[must_use] 
pub fn is_codec_available(fourcc: FourCC) -> bool {
    // In real implementation, would query system for codec support
    matches!(fourcc,
        FourCC::MJPEG | FourCC::H264 | FourCC::H265 |
        FourCC::VP8 | FourCC::VP9 | FourCC::MP4V |
        FourCC::XVID | FourCC::I420
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MatDepth, types::Scalar};

    #[test]
    fn test_video_writer() {
        let mut writer = VideoWriter::new(
            "test_output.mp4",
            FourCC::H264,
            30.0,
            640,
            480,
            true,
        )
        .unwrap();

        assert!(writer.is_opened());

        let frame = Mat::new_with_default(480, 640, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        writer.write(&frame).unwrap();
        assert_eq!(writer.frame_count(), 1);

        writer.release().unwrap();
        assert!(!writer.is_opened());
    }

    #[test]
    fn test_fourcc() {
        let fourcc = FourCC::H264;
        let code = fourcc.to_int();
        let recovered = FourCC::from_int(code);

        match recovered {
            FourCC::Custom(bytes) => {
                assert_eq!(bytes, [b'H', b'2', b'6', b'4']);
            }
            _ => {}
        }
    }

    #[test]
    fn test_fourcc_from_str() {
        let fourcc = FourCC::from_str("MJPG").unwrap();
        match fourcc {
            FourCC::Custom(bytes) => {
                assert_eq!(bytes, [b'M', b'J', b'P', b'G']);
            }
            _ => panic!("Expected Custom FourCC"),
        }
    }

    #[test]
    fn test_codec_availability() {
        assert!(is_codec_available(FourCC::H264));
        assert!(is_codec_available(FourCC::MJPEG));
    }
}
