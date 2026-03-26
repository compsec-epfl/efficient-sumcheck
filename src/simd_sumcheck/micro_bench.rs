/// Quick micro-benchmark to isolate multiply cost vs allocation overhead.
///
/// Run with: cargo test --release --lib micro_bench -- --nocapture

#[cfg(test)]
mod tests {
    use crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    use crate::simd_fields::SimdBaseField;
    use crate::tests::F64;
    use ark_ff::{Field, PrimeField, UniformRand};
    use ark_std::test_rng;

    #[test]
    fn micro_bench_multiply() {
        let n = 1 << 20; // 1M elements
        let iters = 5;

        let mut rng = test_rng();
        let a_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let b_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let a_raw: Vec<u64> = a_ff.iter().map(|f| f.into_bigint().0[0]).collect();
        let b_raw: Vec<u64> = b_ff.iter().map(|f| f.into_bigint().0[0]).collect();

        // Warm up
        let mut sink = 0u64;

        // === Arkworks multiply ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= (a_ff[i] * b_ff[i]).into_bigint().0[0];
            }
        }
        let arkworks_time = start.elapsed();
        println!("Arkworks mul: {:?} ({} muls)", arkworks_time, n * iters);
        println!(
            "  per mul: {:.1}ns",
            arkworks_time.as_nanos() as f64 / (n * iters) as f64
        );

        // === Our Goldilocks scalar multiply ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= GoldilocksNeon::scalar_mul(a_raw[i], b_raw[i]);
            }
        }
        let goldilocks_time = start.elapsed();
        println!(
            "Goldilocks scalar mul: {:?} ({} muls)",
            goldilocks_time,
            n * iters
        );
        println!(
            "  per mul: {:.1}ns",
            goldilocks_time.as_nanos() as f64 / (n * iters) as f64
        );

        // === Montgomery Goldilocks scalar multiply ===
        let a_mont: Vec<u64> = a_ff.iter().map(|f| (f.0).0[0]).collect();
        let b_mont: Vec<u64> = b_ff.iter().map(|f| (f.0).0[0]).collect();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= crate::simd_fields::goldilocks::MontGoldilocksSIMD::scalar_mul(
                    a_mont[i], b_mont[i],
                );
            }
        }
        let mont_time = start.elapsed();
        println!(
            "Montgomery scalar mul: {:?} ({} muls)",
            mont_time,
            n * iters
        );
        println!(
            "  per mul: {:.1}ns",
            mont_time.as_nanos() as f64 / (n * iters) as f64
        );

        // === Arkworks add ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= (a_ff[i] + b_ff[i]).into_bigint().0[0];
            }
        }
        let arkworks_add_time = start.elapsed();
        println!("Arkworks add: {:?}", arkworks_add_time);
        println!(
            "  per add: {:.1}ns",
            arkworks_add_time.as_nanos() as f64 / (n * iters) as f64
        );

        // === Our Goldilocks scalar add ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= GoldilocksNeon::scalar_add(a_raw[i], b_raw[i]);
            }
        }
        let goldilocks_add_time = start.elapsed();
        println!("Goldilocks scalar add: {:?}", goldilocks_add_time);
        println!(
            "  per add: {:.1}ns",
            goldilocks_add_time.as_nanos() as f64 / (n * iters) as f64
        );

        // === Vec allocation test ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let v: Vec<u64> = vec![0u64; n];
            sink ^= v[0];
        }
        let alloc_time = start.elapsed();
        println!("Vec alloc ({}): {:?}", n, alloc_time);
        println!(
            "  per alloc: {:.1}ms",
            alloc_time.as_millis() as f64 / iters as f64
        );

        // Prevent optimization
        assert_ne!(sink, u64::MAX - 1);
    }
}
