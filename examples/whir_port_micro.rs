//! Quick microbench: effsc SIMD `inner_product_sumcheck` vs the WHIR port
//! (faithful) vs the WHIR port with fused fold+compute (`whir_sumcheck_fused`).
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+avx512ifma" \
//!     cargo run --release --example whir_port_micro
//!
//! Notes:
//! - One sample per size is a smoke comparison, not a rigorous bench. Expect
//!   ~10% run-to-run noise.
//! - Each variant is called on freshly-cloned inputs so the timings aren't
//!   biased by cached-state differences.
//! - For F64Ext3, the effsc run uses `inner_product_sumcheck::<F64Ext3, F64Ext3>`
//!   (both a/b and challenges in the extension) to match the WHIR port's
//!   monomorphic signature. The canonical "cross-field" setting (base-field
//!   evals, extension-field challenges) isn't covered here yet — the WHIR port
//!   doesn't support it.

use std::time::Instant;

use ark_ff::Field;
use ark_std::rand::{rngs::StdRng, SeedableRng};

use efficient_sumcheck::tests::{F64Ext3, F64};
use efficient_sumcheck::transcript::SanityTranscript;
use efficient_sumcheck::{inner_product_sumcheck, whir_sumcheck, whir_sumcheck_fused};

const SEED: u64 = 0xA110C8ED;

fn gen_inputs<F: Field>(n: usize) -> (Vec<F>, Vec<F>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let a: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
    let b: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
    (a, b)
}

fn time_effsc<F: Field + From<F>>(a: &[F], b: &[F]) -> f64 {
    let mut f = a.to_vec();
    let mut g = b.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = inner_product_sumcheck::<F, F>(&mut f, &mut g, &mut t);
    start.elapsed().as_secs_f64()
}

fn time_whir_port<F: Field>(a: &[F], b: &[F]) -> f64 {
    let mut f = a.to_vec();
    let mut g = b.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = whir_sumcheck(&mut f, &mut g, &mut t);
    start.elapsed().as_secs_f64()
}

fn time_whir_fused<F: Field>(a: &[F], b: &[F]) -> f64 {
    let mut f = a.to_vec();
    let mut g = b.to_vec();
    let mut trng = StdRng::seed_from_u64(SEED);
    let mut t = SanityTranscript::new(&mut trng);
    let start = Instant::now();
    let _ = whir_sumcheck_fused(&mut f, &mut g, &mut t);
    start.elapsed().as_secs_f64()
}

fn run_section<F: Field + From<F>>(name: &str, sizes: &[u32]) {
    println!("\n== {name} ==");
    println!(
        "{:>6} {:>14} {:>14} {:>14} {:>10} {:>10}",
        "log2 n", "effsc (SIMD)", "whir port", "whir fused", "port/effsc", "fused/port"
    );
    println!("{}", "-".repeat(78));
    for &log2n in sizes {
        let n = 1usize << log2n;
        let (a, b) = gen_inputs::<F>(n);

        // Warm up the allocator/caches once so the first-size timing isn't
        // penalised vs later sizes.
        let _ = time_whir_port(&a[..(n.min(1 << 16))], &b[..(n.min(1 << 16))]);

        let effsc = time_effsc::<F>(&a, &b);
        let whir = time_whir_port::<F>(&a, &b);
        let fused = time_whir_fused::<F>(&a, &b);
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

fn main() {
    run_section::<F64>("Goldilocks (F64, 8 B)", &[20, 21, 22, 23, 24]);
    // F64Ext3 is 24 B/element; cap at 2^22 to stay under ~300 MiB per vector.
    run_section::<F64Ext3>("Goldilocks³ (F64Ext3, 24 B)", &[18, 19, 20, 21, 22]);
}
