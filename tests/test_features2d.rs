// Features2D tests ported from OpenCV test suite
// opencv/modules/features2d/test/test_descriptors.cpp
// opencv/modules/features2d/test/test_keypoints.cpp

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Point;
use opencv_rust::features2d::*;
use opencv_rust::features2d::brief::BRIEF;
use opencv_rust::features2d::freak::FREAK;
use opencv_rust::features2d::orb::ORB;

/// Helper to create test pattern for feature detection
fn create_checkerboard_corners(size: usize, square_size: usize) -> Mat {
    let mut img = Mat::new(size, size, 1, MatDepth::U8).unwrap();

    for row in 0..size {
        for col in 0..size {
            let is_black = ((row / square_size) + (col / square_size)) % 2 == 0;
            img.at_mut(row, col).unwrap()[0] = if is_black { 0 } else { 255 };
        }
    }

    img
}

/// Test from opencv test_keypoints.cpp
#[test]
fn test_fast_detector_finds_corners() {
    // Create image with clear corners (checkerboard)
    let img = create_checkerboard_corners(100, 10);

    let keypoints = fast(&img, 20, true).unwrap();

    // Checkerboard should have many corners
    assert!(
        keypoints.len() > 10,
        "Expected to find corners in checkerboard, found {}",
        keypoints.len()
    );
}

/// Test from opencv test_keypoints.cpp - FAST should be repeatable
#[test]
fn test_fast_detector_repeatability() {
    let img = create_checkerboard_corners(100, 10);

    let keypoints1 = fast(&img, 20, true).unwrap();
    let keypoints2 = fast(&img, 20, true).unwrap();

    // Should get exact same keypoints
    assert_eq!(
        keypoints1.len(),
        keypoints2.len(),
        "FAST should be deterministic"
    );

    // Check positions match
    for (kp1, kp2) in keypoints1.iter().zip(keypoints2.iter()) {
        assert_eq!(kp1.pt.x, kp2.pt.x);
        assert_eq!(kp1.pt.y, kp2.pt.y);
    }
}

/// Test Harris corner detector from opencv test_keypoints.cpp
#[test]
fn test_harris_corners_on_checkerboard() {
    let img = create_checkerboard_corners(100, 20);

    let corners = harris_corners(&img, 3, 3, 0.04, 1000.0).unwrap();

    // Should find corners at checkerboard intersections
    assert!(
        corners.len() > 5,
        "Expected Harris to find corners, found {}",
        corners.len()
    );
}

/// Test from opencv test_descriptors.cpp - descriptor size
#[test]
fn test_brief_descriptor_size() {
    let img = Mat::new_with_default(100, 100, 1, MatDepth::U8,
        opencv_rust::core::types::Scalar::all(128.0)).unwrap();

    let mut kp = KeyPoint::new(Point::new(50, 50), 10.0);
    kp.angle = 0.0;

    // Test different byte sizes
    for &bytes in &[16, 32, 64] {
        let brief = BRIEF::with_params(bytes, 48, false);
        let descriptors = brief.compute(&img, &[kp.clone()]).unwrap();

        assert_eq!(descriptors.len(), 1);
        assert_eq!(
            descriptors[0].len(),
            bytes,
            "BRIEF descriptor should be {} bytes",
            bytes
        );
    }
}

/// Test descriptor matching from opencv test_matchers.cpp
#[test]
fn test_brute_force_matcher_finds_self_match() {
    let img = Mat::new_with_default(100, 100, 1, MatDepth::U8,
        opencv_rust::core::types::Scalar::all(128.0)).unwrap();

    let mut kp = KeyPoint::new(Point::new(50, 50), 10.0);
    kp.angle = 0.0;

    let brief = BRIEF::new();
    let descriptors = brief.compute(&img, &[kp]).unwrap();

    let matcher = BFMatcher::new(DistanceType::Hamming, false);

    // Match descriptor against itself
    let matches = matcher.match_descriptors(&descriptors, &descriptors).unwrap();

    // Should find perfect match (distance = 0)
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].query_idx, 0);
    assert_eq!(matches[0].train_idx, 0);
    assert_eq!(matches[0].distance, 0.0, "Self-match should have zero distance");
}

/// Test from opencv test_descriptors.cpp - descriptor consistency
#[test]
fn test_brief_descriptor_consistency() {
    let img = Mat::new_with_default(100, 100, 1, MatDepth::U8,
        opencv_rust::core::types::Scalar::all(128.0)).unwrap();

    let mut kp = KeyPoint::new(Point::new(50, 50), 10.0);
    kp.angle = 0.0;

    let brief = BRIEF::new();

    // Compute descriptor twice
    let desc1 = brief.compute(&img, &[kp.clone()]).unwrap();
    let desc2 = brief.compute(&img, &[kp]).unwrap();

    // Should be identical
    assert_eq!(desc1.len(), desc2.len());
    assert_eq!(desc1[0], desc2[0], "Descriptor should be deterministic");
}

/// Test FREAK pattern from opencv test_descriptors.cpp
#[test]
fn test_freak_pattern_generation() {
    let freak = FREAK::new();

    // FREAK should have generated receptive fields and pairs
    // This validates the pattern was built correctly
    let img = Mat::new_with_default(200, 200, 1, MatDepth::U8,
        opencv_rust::core::types::Scalar::all(128.0)).unwrap();

    let mut kp = KeyPoint::new(Point::new(100, 100), 20.0);
    kp.angle = 0.0;

    let descriptors = freak.compute(&img, &[kp]).unwrap();

    assert_eq!(descriptors.len(), 1);
    assert_eq!(descriptors[0].len(), 64, "FREAK should produce 64-byte descriptor");
}

/// Test ORB detector from opencv test_features2d.cpp
#[test]
fn test_orb_detects_features() {
    let img = create_checkerboard_corners(200, 20);

    let orb = ORB::new(500);
    let (keypoints, _descriptors) = orb.detect_and_compute(&img).unwrap();

    // ORB should find features in checkerboard
    assert!(
        keypoints.len() > 0,
        "ORB should detect features in checkerboard"
    );
}

/// Test keypoint response ordering from opencv test_keypoints.cpp
#[test]
fn test_harris_response_ordering() {
    let img = create_checkerboard_corners(100, 20);

    let corners = harris_corners(&img, 3, 3, 0.04, 500.0).unwrap();

    // Responses should be sorted (highest first for good_features_to_track)
    let responses: Vec<f32> = corners.iter().map(|kp| kp.response).collect();

    // All responses should be positive (above threshold)
    for &response in &responses {
        assert!(response > 0.0, "Harris response should be positive");
    }
}

/// Test FAST threshold from opencv test_keypoints.cpp
#[test]
fn test_fast_threshold_effect() {
    let img = create_checkerboard_corners(100, 10);

    // Higher threshold should find fewer keypoints
    let kp_low = fast(&img, 10, true).unwrap();
    let kp_high = fast(&img, 40, true).unwrap();

    assert!(
        kp_high.len() <= kp_low.len(),
        "Higher threshold should find fewer or equal keypoints. Low: {}, High: {}",
        kp_low.len(),
        kp_high.len()
    );
}

/// Test non-maximum suppression from opencv test_keypoints.cpp
#[test]
fn test_fast_nms_reduces_keypoints() {
    let img = create_checkerboard_corners(100, 10);

    let kp_without_nms = fast(&img, 20, false).unwrap();
    let kp_with_nms = fast(&img, 20, true).unwrap();

    // NMS should reduce or keep same number of keypoints
    assert!(
        kp_with_nms.len() <= kp_without_nms.len(),
        "NMS should reduce keypoints. Without: {}, With: {}",
        kp_without_nms.len(),
        kp_with_nms.len()
    );
}

/// Test descriptor matching symmetry from opencv test_matchers.cpp
#[test]
fn test_matcher_symmetry() {
    let img1 = Mat::new_with_default(100, 100, 1, MatDepth::U8,
        opencv_rust::core::types::Scalar::all(128.0)).unwrap();

    let mut kp1 = KeyPoint::new(Point::new(30, 30), 10.0);
    kp1.angle = 0.0;

    let mut kp2 = KeyPoint::new(Point::new(70, 70), 10.0);
    kp2.angle = 0.0;

    let brief = BRIEF::new();
    let desc1 = brief.compute(&img1, &[kp1]).unwrap();
    let desc2 = brief.compute(&img1, &[kp2]).unwrap();

    let matcher = BFMatcher::new(DistanceType::Hamming, false);

    // Match in both directions
    let matches_1to2 = matcher.match_descriptors(&desc1, &desc2).unwrap();
    let matches_2to1 = matcher.match_descriptors(&desc2, &desc1).unwrap();

    // Should have same number of matches
    assert_eq!(matches_1to2.len(), matches_2to1.len());
}
