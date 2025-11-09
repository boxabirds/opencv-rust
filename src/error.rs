use thiserror::Error;

/// OpenCV error types
#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid image format: {0}")]
    InvalidFormat(String),

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Out of range: {0}")]
    OutOfRange(String),
}

pub type Result<T> = std::result::Result<T, Error>;
