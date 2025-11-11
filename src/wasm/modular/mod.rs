//! Modular WASM bindings structure
//!
//! This module provides the same WASM API as the monolithic mod.rs,
//! but organized into logical submodules for better maintainability.
//!
//! Enable with: cargo build --target wasm32-unknown-unknown --features wasm,wasm_modular

pub mod filtering;
pub mod morphology;
pub mod color;
pub mod edge_detection;
pub mod threshold;
pub mod contour;
pub mod features;
pub mod transform;
pub mod drawing;
pub mod histogram;
pub mod arithmetic;
pub mod bitwise;
pub mod tracking;
pub mod denoising;
pub mod stereo;
pub mod ml;
pub mod stitching;
pub mod special;

// Re-export all filtering functions
#[cfg(target_arch = "wasm32")]
pub use filtering::{
    gaussian_blur_wasm,
    blur_wasm,
    box_blur_wasm,
    median_blur_wasm,
    bilateral_filter_wasm,
    laplacian_wasm,
    guided_filter_wasm,
    gabor_filter_wasm,
    filter2d_wasm,
};

// Re-export all morphology functions
#[cfg(target_arch = "wasm32")]
pub use morphology::{
    erode_wasm,
    dilate_wasm,
    morphology_opening_wasm,
    morphology_closing_wasm,
    morphology_gradient_wasm,
    morphology_top_hat_wasm,
    morphology_black_hat_wasm,
    morphology_tophat_wasm,
    morphology_blackhat_wasm,
};

// Re-export all color conversion functions
#[cfg(target_arch = "wasm32")]
pub use color::{
    cvt_color_gray_wasm,
    cvt_color_hsv_wasm,
    cvt_color_lab_wasm,
    cvt_color_ycrcb_wasm,
    rgb_to_gray_wasm,
    rgb_to_hsv_wasm,
    rgb_to_lab_wasm,
    rgb_to_ycrcb_wasm,
    hsv_to_rgb_wasm,
    lab_to_rgb_wasm,
    ycrcb_to_rgb_wasm,
};

// Re-export all edge detection functions
#[cfg(target_arch = "wasm32")]
pub use edge_detection::{
    canny_wasm,
    sobel_wasm,
    scharr_wasm,
    hough_lines_wasm,
    hough_lines_p_wasm,
    hough_circles_wasm,
};

// Re-export all threshold functions
#[cfg(target_arch = "wasm32")]
pub use threshold::{
    threshold_wasm,
    adaptive_threshold_wasm,
    in_range_wasm,
};

// Re-export all contour analysis functions
#[cfg(target_arch = "wasm32")]
pub use contour::{
    find_contours_wasm,
    bounding_rect_wasm,
    contour_area_wasm,
    arc_length_wasm,
    approx_poly_dp_wasm,
    moments_wasm,
    min_enclosing_circle_wasm,
    convex_hull_wasm,
    hu_moments_wasm,
    match_shapes_wasm,
    watershed_wasm,
};

// Re-export all feature detection functions
#[cfg(target_arch = "wasm32")]
pub use features::{
    harris_corners_wasm,
    good_features_to_track_wasm,
    fast_wasm,
    sift_wasm,
    orb_wasm,
    brisk_wasm,
    akaze_wasm,
    kaze_wasm,
};

// Re-export all transform functions
#[cfg(target_arch = "wasm32")]
pub use transform::{
    flip_wasm,
    rotate_wasm,
    warp_affine_wasm,
    warp_perspective_wasm,
    pyr_down_wasm,
    pyr_up_wasm,
    get_rotation_matrix_2d_wasm,
    distance_transform_wasm,
    remap_wasm,
};

// Re-export all drawing functions
#[cfg(target_arch = "wasm32")]
pub use drawing::{
    draw_line_wasm,
    draw_rectangle_wasm,
    draw_circle_wasm,
    draw_ellipse_wasm,
    draw_polylines_wasm,
    put_text_wasm,
};

// Re-export all histogram functions
#[cfg(target_arch = "wasm32")]
pub use histogram::{
    equalize_histogram_wasm,
    calc_histogram_wasm,
    normalize_histogram_wasm,
    compare_histograms_wasm,
    back_projection_wasm,
};

// Re-export all arithmetic functions
#[cfg(target_arch = "wasm32")]
pub use arithmetic::{
    add_wasm,
    subtract_wasm,
    multiply_wasm,
    min_wasm,
    max_wasm,
    exp_wasm,
    log_wasm,
    sqrt_wasm,
    pow_wasm,
    add_weighted_wasm,
    convert_scale_wasm,
    absdiff_wasm,
    normalize_wasm,
};

// Re-export all bitwise functions
#[cfg(target_arch = "wasm32")]
pub use bitwise::{
    bitwise_not_wasm,
    bitwise_and_wasm,
    bitwise_or_wasm,
    bitwise_xor_wasm,
};

// Re-export all tracking functions
#[cfg(target_arch = "wasm32")]
pub use tracking::{
    farneback_optical_flow_wasm,
    meanshift_tracker_wasm,
    camshift_tracker_wasm,
    mosse_tracker_wasm,
    csrt_tracker_wasm,
    bg_subtractor_mog2_wasm,
    bg_subtractor_knn_wasm,
    hog_descriptor_wasm,
};

// Re-export all denoising functions
#[cfg(target_arch = "wasm32")]
pub use denoising::{
    nlm_denoising_wasm,
    fast_nl_means_wasm,
    anisotropic_diffusion_wasm,
};

// Re-export all stereo vision and calibration functions
#[cfg(target_arch = "wasm32")]
pub use stereo::{
    calibrate_camera_wasm,
    fisheye_calibration_wasm,
    solve_pnp_wasm,
    stereo_calibration_wasm,
    compute_disparity_wasm,
    stereo_rectification_wasm,
};

// Re-export all machine learning functions
#[cfg(target_arch = "wasm32")]
pub use ml::{
    svm_classifier_wasm,
    decision_tree_wasm,
    random_forest_wasm,
    knn_wasm,
    neural_network_wasm,
    cascade_classifier_wasm,
    load_network_wasm,
    blob_from_image_wasm,
};

// Re-export all stitching functions
#[cfg(target_arch = "wasm32")]
pub use stitching::{
    panorama_stitcher_wasm,
    feather_blender_wasm,
    multiband_blender_wasm,
};

// Re-export all special functions
#[cfg(target_arch = "wasm32")]
pub use special::{
    detect_aruco_wasm,
    detect_qr_wasm,
    log_filter_wasm,
    inpaint_wasm,
    kmeans_wasm,
    tonemap_drago_wasm,
    tonemap_reinhard_wasm,
    find_homography_wasm,
    brute_force_matcher_wasm,
    super_resolution_wasm,
    merge_debevec_wasm,
    gradient_magnitude_wasm,
    integral_image_wasm,
    lut_wasm,
    split_channels_wasm,
    merge_channels_wasm,
};
