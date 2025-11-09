use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use opencv_rust::prelude::*;
use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::{Scalar, Size};
use opencv_rust::imgproc::*;
use opencv_rust::features2d::*;
use opencv_rust::ml::*;

fn bench_mat_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mat Creation");

    for size in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("new", size), size, |b, &size| {
            b.iter(|| {
                Mat::new(black_box(size), black_box(size), black_box(3), black_box(MatDepth::U8))
            })
        });

        group.bench_with_input(BenchmarkId::new("with_default", size), size, |b, &size| {
            b.iter(|| {
                Mat::new_with_default(
                    black_box(size),
                    black_box(size),
                    black_box(3),
                    black_box(MatDepth::U8),
                    black_box(Scalar::all(128.0))
                )
            })
        });
    }

    group.finish();
}

fn bench_mat_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mat Access");
    let mat = Mat::new_with_default(500, 500, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

    group.bench_function("sequential_read", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for row in 0..mat.rows() {
                for col in 0..mat.cols() {
                    let pixel = mat.at(black_box(row), black_box(col)).unwrap();
                    sum += pixel[0] as u64;
                }
            }
            black_box(sum)
        })
    });

    let mut mat_mut = Mat::new_with_default(500, 500, 3, MatDepth::U8, Scalar::all(0.0)).unwrap();
    group.bench_function("sequential_write", |b| {
        b.iter(|| {
            for row in 0..mat_mut.rows() {
                for col in 0..mat_mut.cols() {
                    let pixel = mat_mut.at_mut(black_box(row), black_box(col)).unwrap();
                    pixel[0] = black_box(128);
                }
            }
        })
    });

    group.finish();
}

fn bench_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gaussian Blur");
    let img = Mat::new_with_default(512, 512, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

    for ksize in [3, 5, 7, 11].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(ksize), ksize, |b, &ksize| {
            b.iter(|| {
                let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
                gaussian_blur(&img, &mut dst, Size::new(ksize, ksize), 1.5).unwrap();
                black_box(dst)
            })
        });
    }

    group.finish();
}

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Resize");
    let img = Mat::new_with_default(640, 480, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

    let sizes = [
        ("downscale_2x", 320, 240),
        ("downscale_4x", 160, 120),
        ("upscale_2x", 1280, 960),
    ];

    for &(name, width, height) in sizes.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &(width, height), |b, (w, h)| {
            b.iter(|| {
                let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
                resize(&img, &mut dst, Size::new(*w, *h), InterpolationFlag::Linear).unwrap();
                black_box(dst)
            })
        });
    }

    group.finish();
}

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("Threshold");
    let img = Mat::new_with_default(512, 512, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

    for thresh_type in [ThresholdType::Binary, ThresholdType::BinaryInv, ThresholdType::Trunc].iter() {
        let name = format!("{:?}", thresh_type);
        group.bench_with_input(BenchmarkId::from_parameter(&name), thresh_type, |b, &ttype| {
            b.iter(|| {
                let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
                threshold(&img, &mut dst, 127.0, 255.0, ttype).unwrap();
                black_box(dst)
            })
        });
    }

    group.finish();
}

fn bench_canny(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canny Edge Detection");
    let img = Mat::new_with_default(512, 512, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

    group.bench_function("canny_default", |b| {
        b.iter(|| {
            let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
            canny(&img, &mut dst, 50.0, 150.0).unwrap();
            black_box(dst)
        })
    });

    group.finish();
}

fn bench_harris_corners(c: &mut Criterion) {
    let mut group = c.benchmark_group("Harris Corner Detection");

    // Create test pattern with some features
    let mut img = Mat::new(256, 256, 1, MatDepth::U8).unwrap();
    for row in 0..256 {
        for col in 0..256 {
            let val = if (row / 32 + col / 32) % 2 == 0 { 0 } else { 255 };
            img.at_mut(row, col).unwrap()[0] = val;
        }
    }

    group.bench_function("harris_256x256", |b| {
        b.iter(|| {
            let corners = harris_corners(&img, 3, 3, 0.04, 1000.0).unwrap();
            black_box(corners)
        })
    });

    group.finish();
}

fn bench_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("FAST Feature Detection");

    // Create test pattern
    let mut img = Mat::new(256, 256, 1, MatDepth::U8).unwrap();
    for row in 0..256 {
        for col in 0..256 {
            img.at_mut(row, col).unwrap()[0] = 50;
        }
    }

    // Add some bright spots
    for i in (20..240).step_by(30) {
        for j in (20..240).step_by(30) {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let row = (i as i32 + dy) as usize;
                    let col = (j as i32 + dx) as usize;
                    img.at_mut(row, col).unwrap()[0] = 255;
                }
            }
        }
    }

    group.bench_function("fast_without_nms", |b| {
        b.iter(|| {
            let keypoints = fast(&img, 20, false).unwrap();
            black_box(keypoints)
        })
    });

    group.bench_function("fast_with_nms", |b| {
        b.iter(|| {
            let keypoints = fast(&img, 20, true).unwrap();
            black_box(keypoints)
        })
    });

    group.finish();
}

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("K-Means Clustering");

    // Create sample data
    let mut data = Vec::new();
    for i in 0..100 {
        data.push(vec![i as f64 / 10.0, (100 - i) as f64 / 10.0]);
    }

    for k in [2, 3, 5].iter() {
        group.bench_with_input(BenchmarkId::new("k", k), k, |b, &k| {
            b.iter(|| {
                let mut labels = vec![0; data.len()];
                let (centers, _) = kmeans(
                    &data,
                    black_box(k),
                    &mut labels,
                    100,
                    1.0,
                    KMeansFlags::PPCenters
                ).unwrap();
                black_box(centers)
            })
        });
    }

    group.finish();
}

fn bench_svm(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVM");

    // Create training data
    let samples = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0],
        vec![6.0, 5.0],
        vec![7.0, 6.0],
        vec![8.0, 6.0],
    ];
    let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

    group.bench_function("train_linear", |b| {
        b.iter(|| {
            let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::Linear);
            svm.c = 1.0;
            svm.train(black_box(&samples), black_box(&labels)).unwrap();
            black_box(svm)
        })
    });

    let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::Linear);
    svm.c = 1.0;
    svm.train(&samples, &labels).unwrap();

    group.bench_function("predict", |b| {
        b.iter(|| {
            let pred = svm.predict(black_box(&vec![2.0, 2.5])).unwrap();
            black_box(pred)
        })
    });

    group.finish();
}

fn bench_decision_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decision Tree");

    let data = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        vec![5.0], vec![6.0], vec![7.0], vec![8.0],
    ];
    let labels = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

    group.bench_function("train", |b| {
        b.iter(|| {
            let mut tree = DecisionTree::classifier().with_max_depth(5);
            tree.train(black_box(&data), black_box(&labels)).unwrap();
            black_box(tree)
        })
    });

    let mut tree = DecisionTree::classifier().with_max_depth(5);
    tree.train(&data, &labels).unwrap();

    group.bench_function("predict", |b| {
        b.iter(|| {
            let pred = tree.predict(black_box(&vec![2.5])).unwrap();
            black_box(pred)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mat_creation,
    bench_mat_access,
    bench_blur,
    bench_resize,
    bench_threshold,
    bench_canny,
    bench_harris_corners,
    bench_fast,
    bench_kmeans,
    bench_svm,
    bench_decision_tree,
);

criterion_main!(benches);
