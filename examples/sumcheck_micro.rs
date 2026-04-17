//! Microbench: multilinear and inner-product sumcheck on Goldilocks and
//! its cubic extensions. Single sample per size — smoke comparison, not a
//! rigorous bench (expect ~10% run-to-run noise).
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+avx512ifma" \
//!     cargo run --release --example sumcheck_micro

use std::time::Instant;

use ark_ff::Field;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use efficient_sumcheck::tests::{F64Ext2, F64Ext3, F64};
use efficient_sumcheck::transcript::SanityTranscript;
use efficient_sumcheck::{inner_product_sumcheck, multilinear_sumcheck};

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

fn time_ml<F: Field>(v: &[F]) -> f64 {
    let mut v = v.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = multilinear_sumcheck(&mut v, &mut t, |_, _| {});
    start.elapsed().as_secs_f64()
}

fn time_ip<F: Field>(a: &[F], b: &[F]) -> f64 {
    let mut f = a.to_vec();
    let mut g = b.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = inner_product_sumcheck(&mut f, &mut g, &mut t, |_, _| {});
    start.elapsed().as_secs_f64()
}

fn run_section<F: Field>(name: &str, sizes: &[u32]) {
    println!("\n== {name} ==");
    println!("{:>6} {:>14} {:>14}", "log2 n", "multilinear", "inner prod");
    println!("{}", "-".repeat(40));
    for &log2n in sizes {
        let n = 1usize << log2n;

        // Warm up allocator/caches once so the first-size timing isn't
        // penalised vs later sizes.
        let warm_n = n.min(1 << 16);
        let warm_v = gen_single::<F>(warm_n);
        let _ = time_ml(&warm_v);

        let v = gen_single::<F>(n);
        let ml = time_ml::<F>(&v);
        drop(v); // free before allocating the IP pair.

        let (a, b) = gen_pair::<F>(n);
        let ip = time_ip::<F>(&a, &b);

        println!("{:>6} {:>11.3} ms {:>11.3} ms", log2n, ml * 1e3, ip * 1e3);
    }
}

fn main() {
    // g1 = Goldilocks (8 B), g2 = Goldilocks² (16 B), g3 = Goldilocks³ (24 B).
    run_section::<F64>("g1: Goldilocks (F64, 8 B)", &[20, 21, 22, 23, 24]);
    run_section::<F64Ext2>("g2: Goldilocks² (F64Ext2, 16 B)", &[20, 21, 22, 23, 24]);
    run_section::<F64Ext3>("g3: Goldilocks³ (F64Ext3, 24 B)", &[20, 21, 22, 23, 24]);
}
