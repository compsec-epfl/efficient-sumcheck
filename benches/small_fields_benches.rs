use ark_std::{test_rng, UniformRand, hint::black_box};
use criterion::{Criterion, criterion_group, criterion_main};
use efficient_sumcheck::tests::{
    M31,
    F64 as Goldilocks,
    SmallM31,
    SmallGoldilocks,
};

fn bench_mul_assign_vec_normal_small_m31(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<SmallM31> = (0..16*1024).map(|_| SmallM31::rand(&mut rng)).collect();
    let b: Vec<SmallM31> = (0..16*1024).map(|_| SmallM31::rand(&mut rng)).collect();

    c.bench_function("mul_assign_vec_normal smallm31", |bencher| {
        bencher.iter(|| {
            for i in 0..a.len() {
                a[i] *= &b[i];
            }
            black_box(&a);
        });
    });
}

fn bench_mul_assign_vec_unrolled_small_m31(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<SmallM31> = (0..16*1024).map(|_| SmallM31::rand(&mut rng)).collect();
    let b: Vec<SmallM31> = (0..16*1024).map(|_| SmallM31::rand(&mut rng)).collect();

    c.bench_function("mul_assign_vec_unrolled smallm31", |bencher| {
        bencher.iter(|| {
            let len = a.len();
            let mut i = 0;
            while i + 7 < len {
                a[i] *= b[i];
                a[i + 1] *= b[i + 1];
                a[i + 2] *= b[i + 2];
                a[i + 3] *= b[i + 3];
                a[i + 4] *= b[i + 4];
                a[i + 5] *= b[i + 5];
                a[i + 6] *= b[i + 6];
                a[i + 7] *= b[i + 7];
                // a[i + 8] *= b[i + 8];
                // a[i + 9] *= b[i + 9];
                // a[i + 10] *= b[i + 10];
                // a[i + 11] *= b[i + 11];
                // a[i + 12] *= b[i + 12];
                // a[i + 13] *= b[i + 13];
                // a[i + 14] *= b[i + 14];
                // a[i + 15] *= b[i + 15];
                i += 8;
            }
            black_box(&a);
        });
    });
}

fn bench_mul_assign_vec_normal_m31(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<M31> = (0..16*1024).map(|_| M31::rand(&mut rng)).collect();
    let b: Vec<M31> = (0..16*1024).map(|_| M31::rand(&mut rng)).collect();

    c.bench_function("mul_assign_vec_normal m31", |bencher| {
        bencher.iter(|| {
            for i in 0..a.len() {
                a[i] *= &b[i];
            }
            black_box(&a);
        });
    });
}

fn bench_mul_assign_vec_unrolled_m31(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut a: Vec<M31> = (0..16*1024).map(|_| M31::rand(&mut rng)).collect();
    let b: Vec<M31> = (0..16*1024).map(|_| M31::rand(&mut rng)).collect();

    c.bench_function("mul_assign_vec_unrolled m31", |bencher| {
        bencher.iter(|| {
            let len = a.len();
            let mut i = 0;
            while i + 7 < len {
                a[i] *= b[i];
                a[i + 1] *= b[i + 1];
                a[i + 2] *= b[i + 2];
                a[i + 3] *= b[i + 3];
                a[i + 4] *= b[i + 4];
                a[i + 5] *= b[i + 5];
                a[i + 6] *= b[i + 6];
                a[i + 7] *= b[i + 7];
                // a[i + 8] *= b[i + 8];
                // a[i + 9] *= b[i + 9];
                // a[i + 10] *= b[i + 10];
                // a[i + 11] *= b[i + 11];
                // a[i + 12] *= b[i + 12];
                // a[i + 13] *= b[i + 13];
                // a[i + 14] *= b[i + 14];
                // a[i + 15] *= b[i + 15];
                i += 8;
            }
            black_box(&a);
        });
    });
}

criterion_group!(
    benches,
    bench_mul_assign_vec_normal_small_m31,
    bench_mul_assign_vec_unrolled_small_m31,
    bench_mul_assign_vec_normal_m31,
    bench_mul_assign_vec_unrolled_m31,
);
criterion_main!(benches);