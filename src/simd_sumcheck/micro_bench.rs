/// Quick micro-benchmark to isolate multiply cost vs allocation overhead.
///
/// Run with: cargo test --release --lib micro_bench -- --nocapture

#[cfg(test)]
mod tests {
    use crate::simd_fields::goldilocks::mont_neon::MontGoldilocksNeon;
    use crate::simd_fields::SimdBaseField;
    use crate::tests::F64;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn micro_bench_multiply() {
        let n = 1 << 20; // 1M elements
        let iters = 5;

        let mut rng = test_rng();
        let a_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let b_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let a_mont: Vec<u64> = a_ff.iter().map(|f| f.value).collect();
        let b_mont: Vec<u64> = b_ff.iter().map(|f| f.value).collect();

        let mut sink = 0u64;

        // === Arkworks multiply ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= (a_ff[i] * b_ff[i]).value;
            }
        }
        let arkworks_time = start.elapsed();
        println!("Arkworks mul: {:?} ({} muls)", arkworks_time, n * iters);
        println!(
            "  per mul: {:.1}ns",
            arkworks_time.as_nanos() as f64 / (n * iters) as f64
        );

        // === Montgomery SIMD scalar multiply ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= MontGoldilocksNeon::scalar_mul(a_mont[i], b_mont[i]);
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
                sink ^= (a_ff[i] + b_ff[i]).value;
            }
        }
        let arkworks_add_time = start.elapsed();
        println!("Arkworks add: {:?}", arkworks_add_time);
        println!(
            "  per add: {:.1}ns",
            arkworks_add_time.as_nanos() as f64 / (n * iters) as f64
        );

        // === Montgomery SIMD scalar add ===
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                sink ^= MontGoldilocksNeon::scalar_add(a_mont[i], b_mont[i]);
            }
        }
        let mont_add_time = start.elapsed();
        println!("Montgomery scalar add: {:?}", mont_add_time);
        println!(
            "  per add: {:.1}ns",
            mont_add_time.as_nanos() as f64 / (n * iters) as f64
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
