//! SIMD-vectorized pairwise evaluation: computes (sum_even, sum_odd).
//!
//! Uses an 8-accumulator unroll for instruction-level parallelism,
//! which is the sweet spot on NEON (saturates the register file without
//! spilling — see "Proof Systems Engineering" for benchmarking methodology).

use crate::simd_fields::SimdBaseField;

/// SIMD-vectorized pairwise evaluate.
///
/// Given `src` = `[f(0), f(1), f(2), f(3), ...]`, computes:
///   sum_even = f(0) + f(2) + f(4) + ...
///   sum_odd  = f(1) + f(3) + f(5) + ...
///
/// Returns `(sum_even, sum_odd)`.
///
/// # Panics
///
/// Panics if `src.len()` is not a multiple of `8 * F::LANES` (the unroll factor).
pub fn evaluate<F: SimdBaseField>(src: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    let lanes = F::LANES;
    let step = 8 * lanes;
    assert!(
        src.len() % step == 0 || src.is_empty(),
        "src.len() ({}) must be a multiple of {} (8 * LANES)",
        src.len(),
        step
    );

    let zero = F::splat(F::ZERO);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;
    let mut acc4 = zero;
    let mut acc5 = zero;
    let mut acc6 = zero;
    let mut acc7 = zero;

    let ptr = src.as_ptr();
    let mut i = 0;

    while i < src.len() {
        unsafe {
            acc0 = F::add(acc0, F::load(ptr.add(i)));
            acc1 = F::add(acc1, F::load(ptr.add(i + lanes)));
            acc2 = F::add(acc2, F::load(ptr.add(i + 2 * lanes)));
            acc3 = F::add(acc3, F::load(ptr.add(i + 3 * lanes)));
            acc4 = F::add(acc4, F::load(ptr.add(i + 4 * lanes)));
            acc5 = F::add(acc5, F::load(ptr.add(i + 5 * lanes)));
            acc6 = F::add(acc6, F::load(ptr.add(i + 6 * lanes)));
            acc7 = F::add(acc7, F::load(ptr.add(i + 7 * lanes)));
        }
        i += step;
    }

    // Combine accumulators in a tree to keep ILP.
    let total = F::add(
        F::add(F::add(acc0, acc1), F::add(acc2, acc3)),
        F::add(F::add(acc4, acc5), F::add(acc6, acc7)),
    );

    // Extract lanes and sum even/odd groups.
    let mut lanes_buf = [F::ZERO; 32];
    debug_assert!(F::LANES <= 16);
    unsafe { F::store(lanes_buf.as_mut_ptr(), total) };

    let mut even_sum = F::ZERO;
    let mut odd_sum = F::ZERO;
    for (j, &val) in lanes_buf.iter().enumerate().take(F::LANES) {
        if j % 2 == 0 {
            even_sum = F::scalar_add(even_sum, val);
        } else {
            odd_sum = F::scalar_add(odd_sum, val);
        }
    }

    (even_sum, odd_sum)
}

/// Parallel SIMD evaluate with chunking for large arrays.
///
/// Splits `src` into chunks, evaluates each in parallel (when the `parallel`
/// feature is enabled), then combines.
#[cfg(feature = "parallel")]
pub fn evaluate_parallel<F: SimdBaseField>(src: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    use rayon::prelude::*;

    let chunk_size: usize = 32_768;
    let lanes = F::LANES;
    let step = 8 * lanes;
    let chunk_size = chunk_size.div_ceil(step) * step;

    if src.len() <= chunk_size {
        let aligned_len = (src.len() / step) * step;
        let (mut even, mut odd) = if aligned_len > 0 {
            evaluate::<F>(&src[..aligned_len])
        } else {
            (F::ZERO, F::ZERO)
        };
        for (i, &val) in src.iter().enumerate().skip(aligned_len) {
            if i % 2 == 0 {
                even = F::scalar_add(even, val);
            } else {
                odd = F::scalar_add(odd, val);
            }
        }
        return (even, odd);
    }

    src.par_chunks(chunk_size)
        .map(|chunk| {
            let aligned_len = (chunk.len() / step) * step;
            if aligned_len == 0 {
                let mut even = F::ZERO;
                let mut odd = F::ZERO;
                for (i, &val) in chunk.iter().enumerate() {
                    if i % 2 == 0 {
                        even = F::scalar_add(even, val);
                    } else {
                        odd = F::scalar_add(odd, val);
                    }
                }
                (even, odd)
            } else {
                let (e, o) = evaluate::<F>(&chunk[..aligned_len]);
                let mut even = e;
                let mut odd = o;
                for (i, &val) in chunk.iter().enumerate().skip(aligned_len) {
                    if i % 2 == 0 {
                        even = F::scalar_add(even, val);
                    } else {
                        odd = F::scalar_add(odd, val);
                    }
                }
                (even, odd)
            }
        })
        .reduce(
            || (F::ZERO, F::ZERO),
            |(e1, o1), (e2, o2)| (F::scalar_add(e1, e2), F::scalar_add(o1, o2)),
        )
}

/// Non-parallel version of evaluate that handles arbitrary lengths.
#[cfg(not(feature = "parallel"))]
pub fn evaluate_parallel<F: SimdBaseField>(src: &[F::Scalar]) -> (F::Scalar, F::Scalar) {
    let lanes = F::LANES;
    let step = 8 * lanes;
    let aligned_len = (src.len() / step) * step;

    let (mut even, mut odd) = if aligned_len > 0 {
        evaluate::<F>(&src[..aligned_len])
    } else {
        (F::ZERO, F::ZERO)
    };

    for i in aligned_len..src.len() {
        if i % 2 == 0 {
            even = F::scalar_add(even, src[i]);
        } else {
            odd = F::scalar_add(odd, src[i]);
        }
    }

    (even, odd)
}

// ── Product evaluate ────────────────────────────────────────────────────────

/// SIMD-vectorized inner product evaluate.
///
/// Given `f` = `[f(0), f(1), f(2), ...]` and `g` = `[g(0), g(1), g(2), ...]`,
/// computes the coefficients `(a, b)` of the degree-2 round polynomial:
///   a = Σ f[2i] * g[2i]                       (even-even products)
///   b = Σ (f[2i] * g[2i+1] + f[2i+1] * g[2i]) (cross-term)
///
/// Uses `load_deinterleaved` + SIMD mul with 4× unrolling.
///
/// `f` and `g` must have the same length, which must be a multiple of
/// `8 * F::LANES` (4× unroll, each loading 2×LANES from each of f and g).
pub fn product_evaluate<F: SimdBaseField>(
    f: &[F::Scalar],
    g: &[F::Scalar],
) -> (F::Scalar, F::Scalar) {
    debug_assert_eq!(f.len(), g.len());
    let n = f.len();
    let lanes = F::LANES;
    // Each iteration processes 2*LANES elements from each array (one deinterleaved load).
    // With 4× unrolling: step = 4 * 2 * LANES = 8 * LANES.
    let step = 8 * lanes;
    let aligned = (n / step) * step;

    let zero = F::splat(F::ZERO);
    let mut acc_a0 = zero;
    let mut acc_a1 = zero;
    let mut acc_a2 = zero;
    let mut acc_a3 = zero;
    let mut acc_b0 = zero;
    let mut acc_b1 = zero;
    let mut acc_b2 = zero;
    let mut acc_b3 = zero;

    let f_ptr = f.as_ptr();
    let g_ptr = g.as_ptr();

    let mut i = 0;
    while i < aligned {
        unsafe {
            // Group 0
            let (fe0, fo0) = F::load_deinterleaved(f_ptr.add(i));
            let (ge0, go0) = F::load_deinterleaved(g_ptr.add(i));
            acc_a0 = F::add(acc_a0, F::mul(fe0, ge0));
            acc_b0 = F::add(acc_b0, F::add(F::mul(fe0, go0), F::mul(fo0, ge0)));

            // Group 1
            let off1 = 2 * lanes;
            let (fe1, fo1) = F::load_deinterleaved(f_ptr.add(i + off1));
            let (ge1, go1) = F::load_deinterleaved(g_ptr.add(i + off1));
            acc_a1 = F::add(acc_a1, F::mul(fe1, ge1));
            acc_b1 = F::add(acc_b1, F::add(F::mul(fe1, go1), F::mul(fo1, ge1)));

            // Group 2
            let off2 = 4 * lanes;
            let (fe2, fo2) = F::load_deinterleaved(f_ptr.add(i + off2));
            let (ge2, go2) = F::load_deinterleaved(g_ptr.add(i + off2));
            acc_a2 = F::add(acc_a2, F::mul(fe2, ge2));
            acc_b2 = F::add(acc_b2, F::add(F::mul(fe2, go2), F::mul(fo2, ge2)));

            // Group 3
            let off3 = 6 * lanes;
            let (fe3, fo3) = F::load_deinterleaved(f_ptr.add(i + off3));
            let (ge3, go3) = F::load_deinterleaved(g_ptr.add(i + off3));
            acc_a3 = F::add(acc_a3, F::mul(fe3, ge3));
            acc_b3 = F::add(acc_b3, F::add(F::mul(fe3, go3), F::mul(fo3, ge3)));
        }
        i += step;
    }

    // Combine accumulators in tree
    let total_a = F::add(F::add(acc_a0, acc_a1), F::add(acc_a2, acc_a3));
    let total_b = F::add(F::add(acc_b0, acc_b1), F::add(acc_b2, acc_b3));

    // Horizontal reduce: sum all lanes into a scalar
    let mut buf = [F::ZERO; 32];
    debug_assert!(lanes <= 32);
    let mut a_sum = F::ZERO;
    let mut b_sum = F::ZERO;
    unsafe { F::store(buf.as_mut_ptr(), total_a) };
    for &val in buf.iter().take(lanes) {
        a_sum = F::scalar_add(a_sum, val);
    }
    unsafe { F::store(buf.as_mut_ptr(), total_b) };
    for &val in buf.iter().take(lanes) {
        b_sum = F::scalar_add(b_sum, val);
    }

    // Scalar tail
    let mut i = aligned;
    while i + 1 < n {
        let fe = f[i];
        let fo = f[i + 1];
        let ge = g[i];
        let go = g[i + 1];
        a_sum = F::scalar_add(a_sum, F::scalar_mul(fe, ge));
        b_sum = F::scalar_add(
            b_sum,
            F::scalar_add(F::scalar_mul(fe, go), F::scalar_mul(fo, ge)),
        );
        i += 2;
    }

    (a_sum, b_sum)
}

/// Parallel SIMD product evaluate with chunking for large arrays.
#[cfg(feature = "parallel")]
pub fn product_evaluate_parallel<F: SimdBaseField>(
    f: &[F::Scalar],
    g: &[F::Scalar],
) -> (F::Scalar, F::Scalar) {
    use rayon::prelude::*;

    debug_assert_eq!(f.len(), g.len());
    let n = f.len();
    let lanes = F::LANES;
    let step = 8 * lanes;
    let chunk_size = 32_768_usize.div_ceil(step) * step;

    if n <= chunk_size {
        return product_evaluate::<F>(f, g);
    }

    // Chunk both f and g in lockstep
    f.par_chunks(chunk_size)
        .zip(g.par_chunks(chunk_size))
        .map(|(fc, gc)| product_evaluate::<F>(fc, gc))
        .reduce(
            || (F::ZERO, F::ZERO),
            |(a1, b1), (a2, b2)| (F::scalar_add(a1, a2), F::scalar_add(b1, b2)),
        )
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
pub fn product_evaluate_parallel<F: SimdBaseField>(
    f: &[F::Scalar],
    g: &[F::Scalar],
) -> (F::Scalar, F::Scalar) {
    product_evaluate::<F>(f, g)
}

// ── Extension field evaluate ────────────────────────────────────────────────

/// SIMD-vectorized pairwise evaluate for extension field elements.
///
/// Given `src` containing `n` extension elements of degree `d` (total
/// `n * d` base field scalars in AoS layout: `[e0_c0, e0_c1, ..., e1_c0, ...]`),
/// computes:
///   sum_even = e0 + e2 + e4 + ...   (component-wise)
///   sum_odd  = e1 + e3 + e5 + ...   (component-wise)
///
/// Returns `(even_components, odd_components)` each of length `d`.
///
/// For degree-1 (base field), use [`evaluate`] instead — it's more optimized.
pub fn ext_evaluate<F: SimdBaseField>(
    src: &[F::Scalar],
    ext_degree: usize,
) -> (Vec<F::Scalar>, Vec<F::Scalar>) {
    let n_elems = src.len() / ext_degree;
    debug_assert_eq!(src.len(), n_elems * ext_degree);

    let lanes = F::LANES;
    let n_pairs = n_elems / 2;
    // Stride in u64s between adjacent extension elements
    let elem_stride = ext_degree;
    // Stride in u64s between even and odd element in a pair
    let pair_stride = 2 * ext_degree;

    let mut even_sums = vec![F::ZERO; ext_degree];
    let mut odd_sums = vec![F::ZERO; ext_degree];

    // Number of SIMD vectors needed to load one extension element
    let vecs_per_elem = ext_degree.div_ceil(lanes);

    if ext_degree >= lanes {
        // Optimized path: each extension element is ≥ 1 SIMD vector.
        // Use 4× unrolling for ILP (processes 4 pairs per outer iteration).
        let unroll = 4;
        let aligned_pairs = (n_pairs / unroll) * unroll;

        let simd_components = ext_degree / lanes;
        // 4 even + 4 odd accumulators, each with simd_components vectors
        let zero = F::splat(F::ZERO);
        let mut even_accs = [[zero; 8]; 4]; // [unroll][max_simd_components]
        let mut odd_accs = [[zero; 8]; 4];
        debug_assert!(simd_components <= 8);

        let ptr = src.as_ptr();
        let mut pair = 0;
        while pair < aligned_pairs {
            for u in 0..unroll {
                let p = pair + u;
                let even_off = p * pair_stride;
                let odd_off = even_off + elem_stride;
                for c in 0..simd_components {
                    unsafe {
                        even_accs[u][c] =
                            F::add(even_accs[u][c], F::load(ptr.add(even_off + c * lanes)));
                        odd_accs[u][c] =
                            F::add(odd_accs[u][c], F::load(ptr.add(odd_off + c * lanes)));
                    }
                }
            }
            pair += unroll;
        }

        // Combine unrolled accumulators
        for u in 1..unroll {
            for c in 0..simd_components {
                even_accs[0][c] = F::add(even_accs[0][c], even_accs[u][c]);
                odd_accs[0][c] = F::add(odd_accs[0][c], odd_accs[u][c]);
            }
        }

        // Tail: remaining pairs (< unroll)
        while pair < n_pairs {
            let even_off = pair * pair_stride;
            let odd_off = even_off + elem_stride;
            for c in 0..simd_components {
                unsafe {
                    even_accs[0][c] =
                        F::add(even_accs[0][c], F::load(ptr.add(even_off + c * lanes)));
                    odd_accs[0][c] = F::add(odd_accs[0][c], F::load(ptr.add(odd_off + c * lanes)));
                }
            }
            pair += 1;
        }

        // Extract SIMD lanes into scalar sums
        let mut buf = [F::ZERO; 32];
        for c in 0..simd_components {
            unsafe { F::store(buf.as_mut_ptr(), even_accs[0][c]) };
            for l in 0..lanes {
                even_sums[c * lanes + l] = F::scalar_add(even_sums[c * lanes + l], buf[l]);
            }
            unsafe { F::store(buf.as_mut_ptr(), odd_accs[0][c]) };
            for l in 0..lanes {
                odd_sums[c * lanes + l] = F::scalar_add(odd_sums[c * lanes + l], buf[l]);
            }
        }

        // Tail components (ext_degree not divisible by lanes)
        let tail_start = simd_components * lanes;
        for p in 0..n_pairs {
            let even_off = p * pair_stride;
            let odd_off = even_off + elem_stride;
            for c in tail_start..ext_degree {
                even_sums[c] = F::scalar_add(even_sums[c], src[even_off + c]);
                odd_sums[c] = F::scalar_add(odd_sums[c], src[odd_off + c]);
            }
        }
    } else {
        // ext_degree < LANES (e.g., degree-2 with AVX-512 LANES=8):
        // Multiple extension elements fit in one SIMD vector.
        // Scalar accumulation — still fast for small n.
        let _ = vecs_per_elem;
        for p in 0..n_pairs {
            let even_off = p * pair_stride;
            let odd_off = even_off + elem_stride;
            for c in 0..ext_degree {
                even_sums[c] = F::scalar_add(even_sums[c], src[even_off + c]);
                odd_sums[c] = F::scalar_add(odd_sums[c], src[odd_off + c]);
            }
        }
    }

    (even_sums, odd_sums)
}

/// Parallel extension evaluate with chunking for large arrays.
#[cfg(feature = "parallel")]
pub fn ext_evaluate_parallel<F: SimdBaseField>(
    src: &[F::Scalar],
    ext_degree: usize,
) -> (Vec<F::Scalar>, Vec<F::Scalar>) {
    use rayon::prelude::*;

    let n_elems = src.len() / ext_degree;
    let pair_stride = 2 * ext_degree;
    let chunk_pairs = 8192_usize;
    let chunk_u64s = chunk_pairs * pair_stride;
    let n_pairs = n_elems / 2;

    if n_pairs <= chunk_pairs {
        return ext_evaluate::<F>(src, ext_degree);
    }

    src.par_chunks(chunk_u64s)
        .map(|chunk| ext_evaluate::<F>(chunk, ext_degree))
        .reduce(
            || (vec![F::ZERO; ext_degree], vec![F::ZERO; ext_degree]),
            |(mut e1, mut o1), (e2, o2)| {
                for i in 0..ext_degree {
                    e1[i] = F::scalar_add(e1[i], e2[i]);
                    o1[i] = F::scalar_add(o1[i], o2[i]);
                }
                (e1, o1)
            },
        )
}

/// Non-parallel fallback.
#[cfg(not(feature = "parallel"))]
pub fn ext_evaluate_parallel<F: SimdBaseField>(
    src: &[F::Scalar],
    ext_degree: usize,
) -> (Vec<F::Scalar>, Vec<F::Scalar>) {
    ext_evaluate::<F>(src, ext_degree)
}

// Note: Extension field REDUCE (multiply by challenge) stays in the generic
// arkworks path. The extension multiply is complex (Karatsuba with base muls)
// and on NEON the base mul is scalar anyway. The SIMD win for extensions is
// in the EVALUATE (addition only). For AVX-512 where base mul is truly
// vectorized, a SIMD extension reduce would help — future work.

#[cfg(test)]
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
mod tests {
    use super::*;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    use crate::simd_fields::goldilocks::avx512::GoldilocksAvx512 as Backend;
    #[cfg(target_arch = "aarch64")]
    use crate::simd_fields::goldilocks::neon::GoldilocksNeon as Backend;
    use crate::tests::{to_mont, F64};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_evaluate_matches_pairwise() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 16;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        // Reference: arkworks pairwise evaluate
        let (expected_even, expected_odd) = pairwise::evaluate(&evals_ff);

        // SIMD evaluate (Montgomery domain)
        let (simd_even, simd_odd) = evaluate::<Backend>(&evals_raw);

        assert_eq!(to_mont(expected_even), simd_even, "even sum mismatch");
        assert_eq!(to_mont(expected_odd), simd_odd, "odd sum mismatch");
    }

    #[test]
    fn test_evaluate_parallel_matches_pairwise() {
        use crate::multilinear::reductions::pairwise;

        let mut rng = test_rng();
        let n = 1 << 20;
        let evals_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let evals_raw: Vec<u64> = evals_ff.iter().map(|f| to_mont(*f)).collect();

        let (expected_even, expected_odd) = pairwise::evaluate(&evals_ff);
        let (simd_even, simd_odd) = evaluate_parallel::<Backend>(&evals_raw);

        assert_eq!(
            to_mont(expected_even),
            simd_even,
            "parallel even sum mismatch"
        );
        assert_eq!(to_mont(expected_odd), simd_odd, "parallel odd sum mismatch");
    }

    #[test]
    fn test_product_evaluate_matches_generic() {
        use crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;

        let mut rng = test_rng();
        let n = 1 << 16;
        let f_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let f_raw: Vec<u64> = f_ff.iter().map(|f| to_mont(*f)).collect();
        let g_raw: Vec<u64> = g_ff.iter().map(|g| to_mont(*g)).collect();

        let (expected_a, expected_b) = pairwise_product_evaluate(&[f_ff.clone(), g_ff.clone()]);

        let (simd_a, simd_b) = product_evaluate::<Backend>(&f_raw, &g_raw);

        assert_eq!(to_mont(expected_a), simd_a, "product a mismatch");
        assert_eq!(to_mont(expected_b), simd_b, "product b mismatch");
    }

    #[test]
    fn test_product_evaluate_parallel_matches_generic() {
        use crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;

        let mut rng = test_rng();
        let n = 1 << 20;
        let f_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g_ff: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let f_raw: Vec<u64> = f_ff.iter().map(|f| to_mont(*f)).collect();
        let g_raw: Vec<u64> = g_ff.iter().map(|g| to_mont(*g)).collect();

        let (expected_a, expected_b) = pairwise_product_evaluate(&[f_ff.clone(), g_ff.clone()]);

        let (simd_a, simd_b) = product_evaluate_parallel::<Backend>(&f_raw, &g_raw);

        assert_eq!(to_mont(expected_a), simd_a, "parallel product a mismatch");
        assert_eq!(to_mont(expected_b), simd_b, "parallel product b mismatch");
    }
}
