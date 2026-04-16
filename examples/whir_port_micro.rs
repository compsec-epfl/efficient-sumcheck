//! Quick microbench: effsc SIMD `inner_product_sumcheck` vs the WHIR port
//! (`whir_sumcheck`), one sample per size, Goldilocks (F64).
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+avx512ifma" \
//!     cargo run --release --example whir_port_micro
//!
//! Notes:
//! - One sample per size is a smoke comparison, not a rigorous bench. Expect
//!   ~10% run-to-run noise.
//! - Both variants are called on freshly-cloned inputs so the timings
//!   aren't biased by cached-state differences.
//! - The inputs for the WHIR port are the same vectors as the effsc run
//!   (WHIR consumes half-split / MSB layout natively — no reorder needed).

use std::time::Instant;

use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use efficient_sumcheck::tests::F64;
use efficient_sumcheck::transcript::SanityTranscript;
use efficient_sumcheck::{inner_product_sumcheck, whir_sumcheck, whir_sumcheck_fused};

const LOG2_SIZES: &[u32] = &[20, 21, 22, 23, 24];
const SEED: u64 = 0xA110C8ED;

fn gen_inputs(n: usize) -> (Vec<F64>, Vec<F64>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let a: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
    let b: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
    (a, b)
}

fn time_effsc(a: &[F64], b: &[F64]) -> f64 {
    let mut f = a.to_vec();
    let mut g = b.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = inner_product_sumcheck::<F64, F64>(&mut f, &mut g, &mut t);
    start.elapsed().as_secs_f64()
}

fn time_whir_port(a: &[F64], b: &[F64]) -> f64 {
    let mut f = a.to_vec();
    let mut g = b.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = whir_sumcheck(&mut f, &mut g, &mut t);
    start.elapsed().as_secs_f64()
}

fn time_whir_fused(a: &[F64], b: &[F64]) -> f64 {
    let mut f = a.to_vec();
    let mut g = b.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = whir_sumcheck_fused(&mut f, &mut g, &mut t);
    start.elapsed().as_secs_f64()
}

fn main() {
    println!(
        "{:>6} {:>14} {:>14} {:>14} {:>10} {:>10}",
        "log2 n", "effsc (SIMD)", "whir port", "whir fused", "port/effsc", "fused/port"
    );
    println!("{}", "-".repeat(78));
    for &log2n in LOG2_SIZES {
        let n = 1usize << log2n;
        let (a, b) = gen_inputs(n);

        // Warm up the allocator/caches once so the first-size timing isn't
        // penalised vs later sizes.
        let _ = time_whir_port(&a[..(n.min(1 << 16))], &b[..(n.min(1 << 16))]);

        let effsc = time_effsc(&a, &b);
        let whir = time_whir_port(&a, &b);
        let fused = time_whir_fused(&a, &b);
        println!(
            "{:>6} {:>11.3} ms {:>11.3} ms {:>11.3} ms {:>9.2}x {:>9.2}x",
            log2n,
            effsc * 1e3,
            whir * 1e3,
            fused * 1e3,
            whir / effsc,
            fused / whir,
        );
    }
}
