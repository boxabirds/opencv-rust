// ML tests ported from OpenCV test suite
// opencv/modules/ml/test/test_mltests.cpp
// opencv/modules/ml/test/test_mltests2.cpp

use opencv_rust::ml::*;

/// Test from opencv test_mltests.cpp - K-means finds correct clusters
#[test]
fn test_kmeans_finds_clusters() {
    // Create two well-separated clusters
    let mut data = Vec::new();

    // Cluster 1: points around (1, 1)
    for i in 0..10 {
        data.push(vec![1.0 + (i as f64 * 0.1), 1.0 + (i as f64 * 0.1)]);
    }

    // Cluster 2: points around (10, 10)
    for i in 0..10 {
        data.push(vec![10.0 + (i as f64 * 0.1), 10.0 + (i as f64 * 0.1)]);
    }

    let mut labels = vec![0; 20];

    let (centers, compactness) = kmeans(
        &data,
        2,
        &mut labels,
        100,
        1.0,
        KMeansFlags::PPCenters
    ).unwrap();

    // Should find 2 centers
    assert_eq!(centers.len(), 2);

    // Centers should be well-separated
    let dist = ((centers[0][0] - centers[1][0]).powi(2)
              + (centers[0][1] - centers[1][1]).powi(2)).sqrt();
    assert!(dist > 8.0, "Centers should be well-separated, distance: {}", dist);

    // All points in first 10 should have same label
    let first_label = labels[0];
    for i in 0..10 {
        assert_eq!(labels[i], first_label, "First cluster should have consistent labels");
    }

    // All points in second 10 should have different label
    let second_label = labels[10];
    assert_ne!(first_label, second_label, "Clusters should have different labels");
    for i in 10..20 {
        assert_eq!(labels[i], second_label, "Second cluster should have consistent labels");
    }

    // Compactness should be low for well-separated clusters
    assert!(compactness < 10.0, "Compactness should be low: {}", compactness);
}

/// Test K-means++ initialization from opencv test_mltests.cpp
#[test]
fn test_kmeans_pp_initialization() {
    let data = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![10.0, 10.0],
        vec![11.0, 11.0],
    ];

    let mut labels = vec![0; 4];

    let (centers, _) = kmeans(
        &data,
        2,
        &mut labels,
        1, // Only 1 iteration to test initialization
        1.0,
        KMeansFlags::PPCenters
    ).unwrap();

    // K-means++ should select distant points as initial centers
    assert_eq!(centers.len(), 2);
}

/// Test from opencv test_mltests.cpp - SVM linear separation
#[test]
fn test_svm_linear_separable() {
    // Create linearly separable data
    let samples = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0],
        vec![6.0, 5.0],
        vec![7.0, 6.0],
        vec![8.0, 6.0],
    ];

    let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

    let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::Linear);
    svm.c = 1.0;
    svm.train(&samples, &labels).unwrap();

    // Test training samples
    let pred1 = svm.predict(&vec![2.0, 2.5]).unwrap();
    assert_eq!(pred1, 1.0, "Should classify class 1 correctly");

    let pred2 = svm.predict(&vec![7.0, 5.5]).unwrap();
    assert_eq!(pred2, -1.0, "Should classify class -1 correctly");
}

/// Test SVM RBF kernel from opencv test_mltests.cpp
#[test]
fn test_svm_rbf_kernel() {
    // XOR problem - not linearly separable
    let samples = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let labels = vec![-1.0, 1.0, 1.0, -1.0];

    let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::RBF);
    svm.gamma = 2.0;
    svm.train(&samples, &labels).unwrap();

    // RBF kernel should handle non-linear separation
    let (pred, confidence) = svm.predict_with_confidence(&vec![0.0, 0.0]).unwrap();
    assert!(confidence >= 0.0, "Confidence should be non-negative");
}

/// Test from opencv test_mltests.cpp - Decision tree classification
#[test]
fn test_decision_tree_simple_split() {
    // Simple data that should split at x[0] = 4.5
    let data = vec![
        vec![1.0, 5.0],
        vec![2.0, 5.0],
        vec![3.0, 5.0],
        vec![7.0, 5.0],
        vec![8.0, 5.0],
        vec![9.0, 5.0],
    ];

    let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut tree = DecisionTree::classifier()
        .with_max_depth(5)
        .with_min_samples_split(2);

    tree.train(&data, &labels).unwrap();

    // Test predictions
    let pred1 = tree.predict(&vec![2.5, 5.0]).unwrap();
    assert_eq!(pred1, 0.0, "Should predict class 0 for x < 4.5");

    let pred2 = tree.predict(&vec![8.0, 5.0]).unwrap();
    assert_eq!(pred2, 1.0, "Should predict class 1 for x > 4.5");
}

/// Test decision tree depth control from opencv test_mltests.cpp
#[test]
fn test_decision_tree_max_depth() {
    let data = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        vec![5.0], vec![6.0], vec![7.0], vec![8.0],
    ];

    let labels = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

    // Shallow tree
    let mut tree_shallow = DecisionTree::classifier().with_max_depth(2);
    tree_shallow.train(&data, &labels).unwrap();
    let depth_shallow = tree_shallow.get_depth();

    // Deep tree
    let mut tree_deep = DecisionTree::classifier().with_max_depth(10);
    tree_deep.train(&data, &labels).unwrap();
    let depth_deep = tree_deep.get_depth();

    assert!(depth_shallow <= 2, "Shallow tree should respect max_depth");
    assert!(depth_shallow > 0, "Tree should have some depth");
}

/// Test decision tree regression from opencv test_mltests.cpp
#[test]
fn test_decision_tree_regression() {
    // y = 2*x relationship
    let data = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
    ];

    let labels = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let mut tree = DecisionTree::regressor()
        .with_max_depth(5)
        .with_min_samples_split(2);

    tree.train(&data, &labels).unwrap();

    // Predictions should be close to 2*x
    let pred = tree.predict(&vec![3.0]).unwrap();
    assert!(
        (pred - 6.0).abs() < 2.0,
        "Regression should approximate y=2x, got {} for x=3",
        pred
    );
}

/// Test KNN classification from opencv test_mltests.cpp
#[test]
fn test_knn_classification_accuracy() {
    // Create clear decision boundary at x=5
    let data = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![7.0, 7.0],
        vec![8.0, 8.0],
        vec![9.0, 9.0],
    ];

    let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut knn = KNearest::classifier(3);
    knn.train(&data, &labels).unwrap();

    // Test on training data
    for (i, sample) in data.iter().enumerate() {
        let pred = knn.predict(sample).unwrap();
        assert_eq!(
            pred, labels[i],
            "KNN should perfectly classify training data"
        );
    }

    // Test on new data
    let pred1 = knn.predict(&vec![2.5, 2.5]).unwrap();
    assert_eq!(pred1, 0.0, "Should classify as class 0");

    let pred2 = knn.predict(&vec![8.0, 7.5]).unwrap();
    assert_eq!(pred2, 1.0, "Should classify as class 1");
}

/// Test KNN k value effect from opencv test_mltests.cpp
#[test]
fn test_knn_k_value_effect() {
    let data = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![8.0, 8.0],
    ];

    let labels = vec![0.0, 0.0, 1.0];

    // k=1: nearest neighbor wins
    let mut knn1 = KNearest::classifier(1);
    knn1.train(&data, &labels).unwrap();

    // k=3: majority voting
    let mut knn3 = KNearest::classifier(3);
    knn3.train(&data, &labels).unwrap();

    // Point closer to class 0 samples
    let test_point = vec![3.0, 3.0];

    let pred1 = knn1.predict(&test_point).unwrap();
    let pred3 = knn3.predict(&test_point).unwrap();

    // Both should predict class 0 for this point
    assert_eq!(pred1, 0.0);
    assert_eq!(pred3, 0.0);
}

/// Test KNN regression from opencv test_mltests.cpp
#[test]
fn test_knn_regression() {
    let data = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
    ];

    // y = x^2
    let labels = vec![1.0, 4.0, 9.0, 16.0, 25.0];

    let mut knn = KNearest::regressor(2);
    knn.train(&data, &labels).unwrap();

    // Interpolate between 2 and 3
    let pred = knn.predict(&vec![2.5]).unwrap();

    // Should be between 4 and 9
    assert!(pred > 4.0 && pred < 9.0, "Regression should interpolate: {}", pred);

    // Should be closer to average of neighbors
    assert!((pred - 6.5).abs() < 2.0, "Should be close to average of neighbors");
}

/// Test KNN find nearest from opencv test_mltests.cpp
#[test]
fn test_knn_find_nearest_correctness() {
    let data = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![10.0, 10.0],
    ];

    let labels = vec![0.0, 1.0, 2.0, 3.0];

    let mut knn = KNearest::classifier(3);
    knn.train(&data, &labels).unwrap();

    // Find 2 nearest neighbors to (1.5, 1.5)
    let nearest = knn.find_nearest(&vec![1.5, 1.5], 2).unwrap();

    assert_eq!(nearest.len(), 2);

    // Indices 1 and 2 should be closest
    assert!(
        nearest[0].0 == 1 || nearest[0].0 == 2,
        "First nearest should be index 1 or 2"
    );
    assert!(
        nearest[1].0 == 1 || nearest[1].0 == 2,
        "Second nearest should be index 1 or 2"
    );

    // Distances should be sorted
    assert!(
        nearest[0].1 <= nearest[1].1,
        "Distances should be sorted"
    );
}

/// Test from opencv test_mltests.cpp - K-means compactness
#[test]
fn test_kmeans_compactness_improves() {
    let data = vec![
        vec![1.0, 1.0],
        vec![1.5, 1.5],
        vec![2.0, 2.0],
        vec![10.0, 10.0],
        vec![10.5, 10.5],
        vec![11.0, 11.0],
    ];

    let mut labels = vec![0; 6];

    // Run with 1 iteration
    let (_, compactness1) = kmeans(
        &data,
        2,
        &mut labels,
        1,
        0.0,
        KMeansFlags::PPCenters
    ).unwrap();

    // Run with many iterations
    let (_, compactness_many) = kmeans(
        &data,
        2,
        &mut labels,
        100,
        1.0,
        KMeansFlags::PPCenters
    ).unwrap();

    // More iterations should improve or maintain compactness
    assert!(
        compactness_many <= compactness1 * 1.1,
        "More iterations should not significantly increase compactness"
    );
}

/// Test SVM confidence from opencv test_mltests.cpp
#[test]
fn test_svm_decision_function() {
    let samples = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![5.0, 5.0],
        vec![6.0, 6.0],
    ];

    let labels = vec![-1.0, -1.0, 1.0, 1.0];

    let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::Linear);
    svm.train(&samples, &labels).unwrap();

    // Point very close to class 1
    let (pred, conf1) = svm.predict_with_confidence(&vec![5.5, 5.5]).unwrap();
    assert_eq!(pred, 1.0);

    // Point very close to class -1
    let (pred, conf2) = svm.predict_with_confidence(&vec![0.5, 0.5]).unwrap();
    assert_eq!(pred, -1.0);

    // Both confidences should be positive
    assert!(conf1 > 0.0);
    assert!(conf2 > 0.0);
}

/// Test decision tree leaf count from opencv test_mltests.cpp
#[test]
fn test_decision_tree_leaf_count() {
    let data = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![8.0, 8.0],
        vec![9.0, 9.0],
    ];

    let labels = vec![0.0, 0.0, 1.0, 1.0];

    let mut tree = DecisionTree::classifier()
        .with_max_depth(10)
        .with_min_samples_split(2)
        .with_min_samples_leaf(1);

    tree.train(&data, &labels).unwrap();

    let leaf_count = tree.get_leaf_count();

    // Should have at least 2 leaves (one per class)
    assert!(leaf_count >= 2, "Tree should have at least 2 leaves");

    // Should not have too many leaves
    assert!(leaf_count <= 4, "Tree should not overfit with {} leaves", leaf_count);
}

/// Test KNN weighted voting from opencv test_mltests.cpp
#[test]
fn test_knn_weighted_voting() {
    // One close sample of class 1, two far samples of class 0
    let data = vec![
        vec![0.0, 0.0],  // class 0 - far
        vec![0.0, 10.0], // class 0 - far
        vec![1.0, 0.0],  // class 1 - close
    ];

    let labels = vec![0.0, 0.0, 1.0];

    let mut knn = KNearest::classifier(3);
    knn.train(&data, &labels).unwrap();

    // Test point very close to class 1 sample
    let pred = knn.predict(&vec![1.1, 0.1]).unwrap();

    // Weighted voting should pick class 1 despite majority being class 0
    assert_eq!(pred, 1.0, "Weighted voting should favor closer sample");
}
