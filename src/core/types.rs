use std::ops::{Add, Sub, Mul};

/// 2D point with integer coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    #[must_use] 
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

/// 2D point with floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2f {
    pub x: f32,
    pub y: f32,
}

impl Point2f {
    #[must_use] 
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// 3D point with floating-point coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3f {
    #[must_use] 
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// 2D size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

impl Size {
    #[must_use] 
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    #[must_use] 
    pub fn area(&self) -> i32 {
        self.width * self.height
    }
}

/// Rectangle defined by top-left corner and size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    #[must_use] 
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    #[must_use] 
    pub fn area(&self) -> i32 {
        self.width * self.height
    }

    #[must_use] 
    pub fn top_left(&self) -> Point {
        Point::new(self.x, self.y)
    }

    #[must_use] 
    pub fn bottom_right(&self) -> Point {
        Point::new(self.x + self.width, self.y + self.height)
    }

    #[must_use] 
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x
            && point.x < self.x + self.width
            && point.y >= self.y
            && point.y < self.y + self.height
    }
}

/// Scalar value (up to 4 channels)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scalar {
    pub val: [f64; 4],
}

impl Scalar {
    #[must_use] 
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Self { val: [v0, v1, v2, v3] }
    }

    #[must_use] 
    pub fn all(v: f64) -> Self {
        Self { val: [v, v, v, v] }
    }

    #[must_use] 
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self {
            val: [f64::from(r), f64::from(g), f64::from(b), 0.0],
        }
    }

    #[must_use] 
    pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            val: [f64::from(r), f64::from(g), f64::from(b), f64::from(a)],
        }
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            val: [
                self.val[0] + other.val[0],
                self.val[1] + other.val[1],
                self.val[2] + other.val[2],
                self.val[3] + other.val[3],
            ],
        }
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            val: [
                self.val[0] - other.val[0],
                self.val[1] - other.val[1],
                self.val[2] - other.val[2],
                self.val[3] - other.val[3],
            ],
        }
    }
}

impl Mul<f64> for Scalar {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            val: [
                self.val[0] * scalar,
                self.val[1] * scalar,
                self.val[2] * scalar,
                self.val[3] * scalar,
            ],
        }
    }
}

/// Color conversion codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorConversionCode {
    BgrToGray,
    RgbToGray,
    BgraToGray,
    RgbaToGray,
    GrayToBgr,
    GrayToRgb,
    BgrToRgb,
    RgbToBgr,
    BgrToHsv,
    RgbToHsv,
    HsvToBgr,
    HsvToRgb,
    BgrToLab,
    RgbToLab,
    LabToBgr,
    LabToRgb,
    BgrToYCrCb,
    RgbToYCrCb,
    YCrCbToBgr,
    YCrCbToRgb,
}

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationFlag {
    Nearest,
    Linear,
    Cubic,
    Area,
    Lanczos4,
}

/// Threshold types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_operations() {
        let p1 = Point::new(10, 20);
        let p2 = Point::new(5, 3);
        let sum = p1 + p2;
        assert_eq!(sum, Point::new(15, 23));
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(10, 10, 100, 100);
        assert!(rect.contains(Point::new(50, 50)));
        assert!(!rect.contains(Point::new(5, 5)));
    }

    #[test]
    fn test_scalar_operations() {
        let s1 = Scalar::all(10.0);
        let s2 = Scalar::all(5.0);
        let sum = s1 + s2;
        assert_eq!(sum.val[0], 15.0);
    }
}
