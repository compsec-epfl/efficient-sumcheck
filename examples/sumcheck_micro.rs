//! Microbench: multilinear and inner-product sumcheck on Goldilocks and
//! its cubic extension. Single sample per size — smoke comparison, not a
//! rigorous bench (expect ~10% run-to-run noise).
//!
//! Run:
//!   cargo run --release --example sumcheck_micro
//!
//! With AVX-512:
//!   RUSTFLAGS="-C target-feature=+avx512ifma" \
//!     cargo run --release --example sumcheck_micro

use std::time::Instant;

use ark_ff::Field;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use effsc::provers::inner_product::InnerProductProver;
use effsc::provers::multilinear::MultilinearProver;
use effsc::runner::sumcheck;
use effsc::tests::{F64Ext2, F64Ext3, F64};
use effsc::transcript::SanityTranscript;

const SEED: u64 = 0xA110C8ED;

fn gen_single<F: Field>(n: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..n).map(|_| F::rand(&mut rng)).collect()
}

fn gen_pair<F: Field>(n: usize) -> (Vec<F>, Vec<F>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let a: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
    let b: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
    (a, b)
}

fn time_ml<F: Field>(v: &[F], num_vars: usize) -> f64 {
    let evals = v.to_vec();
    let mut prover = MultilinearProver::new(evals);
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = sumcheck(&mut prover, num_vars, &mut t, |_, _| {});
    start.elapsed().as_secs_f64()
}

fn time_ip<F: Field>(a: &[F], b: &[F], num_vars: usize) -> f64 {
    let prover_a = a.to_vec();
    let prover_b = b.to_vec();
    let mut prover = InnerProductProver::new(prover_a, prover_b);
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = sumcheck(&mut prover, num_vars, &mut t, |_, _| {});
    start.elapsed().as_secs_f64()
}

fn run_section<F: Field>(name: &str, sizes: &[u32]) {
    println!("\n== {name} ==");
    println!("{:>6} {:>14} {:>14}", "log2 n", "multilinear", "inner prod");
    println!("{}", "-".repeat(40));
    for &log2n in sizes {
        let n = 1usize << log2n;
        let nv = log2n as usize;

        // Warm up allocator/caches.
        let warm_n = n.min(1 << 16);
        let warm_v = gen_single::<F>(warm_n);
        let warm_nv = warm_n.trailing_zeros() as usize;
        let _ = time_ml(&warm_v, warm_nv);

        let v = gen_single::<F>(n);
        let ml = time_ml::<F>(&v, nv);
        drop(v);

        let (a, b) = gen_pair::<F>(n);
        let ip = time_ip::<F>(&a, &b, nv);

        println!("{:>6} {:>11.3} ms {:>11.3} ms", log2n, ml * 1e3, ip * 1e3);
    }
}

fn main() {
    run_section::<F64>("g1: Goldilocks (F64, 8 B)", &[20, 21, 22, 23, 24]);
    run_section::<F64Ext2>("g2: Goldilocks² (F64Ext2, 16 B)", &[20, 21, 22, 23, 24]);
    run_section::<F64Ext3>("g3: Goldilocks³ (F64Ext3, 24 B)", &[20, 21, 22, 23, 24]);
}
