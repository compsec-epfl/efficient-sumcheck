//! SIMD-accelerated field operations.
//!
//! General-purpose primitives that auto-dispatch to SIMD backends for
//! Goldilocks-based fields (base field, degree-2 and degree-3 extensions).
//! Falls back to generic arkworks `Field` operations for other fields.
//!
//! These are not sumcheck-specific — any protocol that does pairwise folding,
//! dot products, or multi-scalar operations can use them.
//!
//! # Example
//!
//! ```text
//! use efficient_sumcheck::simd_ops;
//!
//! let mut evals: Vec<F> = /* ... */;
//! let (s0, s1) = simd_ops::pairwise_sum(&evals);
//! simd_ops::fold(&mut evals, challenge);
//! let dot = simd_ops::inner_product(&f, &g);
//! ```

use ark_ff::Field;

// ─── Pairwise sum ───────────────────────────────────────────────────────────

/// Sum even-indexed and odd-indexed elements.
///
/// Returns `(Σ data[2i], Σ data[2i+1])` for `i = 0..data.len()/2`.
///
/// SIMD-accelerated for Goldilocks base and extension fields.
pub fn pairwise_sum<F: Field>(data: &[F]) -> (F, F) {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    if let Some(result) = crate::simd_sumcheck::dispatch::try_simd_ext_evaluate(data) {
        return result;
    }

    // Generic fallback
    let mut even = F::ZERO;
    let mut odd = F::ZERO;
    for i in (0..data.len()).step_by(2) {
        even += data[i];
        if i + 1 < data.len() {
            odd += data[i + 1];
        }
    }
    (even, odd)
}

// ─── Fold ───────────────────────────────────────────────────────────────────

/// Pairwise fold: `data[i] = data[2i] + challenge * (data[2i+1] - data[2i])`.
///
/// Halves the length of `data`. SIMD-accelerated for Goldilocks-based fields.
pub fn fold<F: Field>(data: &mut Vec<F>, challenge: F) {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    {
        // Try SIMD base field reduce
        if crate::simd_sumcheck::dispatch::try_simd_reduce(data, challenge) {
            return;
        }
        // Try SIMD extension field reduce.
        // On AVX-512: always (8-wide IFMA mul is faster than generic).
        // On NEON: only for small inputs (scalar ext mul is slower than
        // rayon-parallel generic reduce at scale).
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
        if crate::simd_sumcheck::dispatch::try_simd_ext_reduce(data, challenge) {
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if data.len() <= (1 << 17)
            && crate::simd_sumcheck::dispatch::try_simd_ext_reduce(data, challenge)
        {
            return;
        }
    }

    // Generic fallback: uses rayon-parallel reduce via arkworks
    crate::multilinear::reductions::pairwise::reduce_evaluations(data, challenge);
}

/// Fold two vectors in one interleaved pass.
///
/// Equivalent to `fold(f, challenge); fold(g, challenge);` but reads
/// f and g data together for better cache utilization.
///
/// SIMD-accelerated for Goldilocks base field.
pub fn fold_both<F: Field>(f: &mut Vec<F>, g: &mut Vec<F>, challenge: F) {
    debug_assert_eq!(f.len(), g.len());

    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    {
        if let Some(did_it) = try_simd_fold_both(f, g, challenge) {
            if did_it {
                return;
            }
        }
    }

    // Fallback: two separate folds
    fold(f, challenge);
    fold(g, challenge);
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
fn try_simd_fold_both<F: Field>(f: &mut Vec<F>, g: &mut Vec<F>, challenge: F) -> Option<bool> {
    use crate::simd_sumcheck::dispatch::{field_to_u64_pub, is_goldilocks_pub};

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    if is_goldilocks_pub::<F>() {
        // Base field: fused interleaved reduce-both kernel.
        let n = f.len();
        let f_raw: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(f.as_mut_ptr() as *mut u64, n) };
        let g_raw: &mut [u64] =
            unsafe { core::slice::from_raw_parts_mut(g.as_mut_ptr() as *mut u64, n) };
        let chg: u64 = field_to_u64_pub(challenge);

        let new_len =
            crate::simd_sumcheck::reduce::reduce_both_in_place::<Backend>(f_raw, g_raw, chg);
        f.truncate(new_len);
        g.truncate(new_len);
        return Some(true);
    }

    // Ext2/ext3: call ext in-place reduce on f and g directly, sharing the
    // challenge/nonresidue setup. Equivalent to `fold(f); fold(g)` but
    // avoids the re-dispatch through `try_simd_reduce` → `try_simd_ext_reduce`
    // on each call. On AVX-512 these kernels use 8-wide IFMA.
    //
    // NEON note: the existing ext reduce kernels do scalar Karatsuba under
    // the SIMD wrapper (no true vector 64×64 mul). They still help vs the
    // generic arkworks reduce for small inputs, but rayon-parallel generic
    // reduce beats them at scale. Keep AVX-512-only routing here.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    {
        use crate::simd_sumcheck::dispatch::{
            extract_nonresidue_ext2, extract_nonresidue_ext3, is_goldilocks_based_pub,
        };
        if is_goldilocks_based_pub::<F>()
            && core::mem::size_of::<F>()
                == (F::extension_degree() as usize) * core::mem::size_of::<u64>()
        {
            let d = F::extension_degree() as usize;
            if d == 2 {
                let chg_raw: [u64; 2] = unsafe {
                    let ptr = &challenge as *const F as *const u64;
                    [*ptr, *ptr.add(1)]
                };
                let w = extract_nonresidue_ext2::<F, Backend>();

                let n_f = f.len() * d;
                let f_buf: &mut [u64] =
                    unsafe { core::slice::from_raw_parts_mut(f.as_mut_ptr() as *mut u64, n_f) };
                crate::simd_sumcheck::reduce::ext2_reduce_in_place::<Backend>(f_buf, chg_raw, w);

                let n_g = g.len() * d;
                let g_buf: &mut [u64] =
                    unsafe { core::slice::from_raw_parts_mut(g.as_mut_ptr() as *mut u64, n_g) };
                crate::simd_sumcheck::reduce::ext2_reduce_in_place::<Backend>(g_buf, chg_raw, w);

                f.truncate(f.len() / 2);
                g.truncate(g.len() / 2);
                return Some(true);
            }
            if d == 3 {
                let chg_raw: [u64; 3] = unsafe {
                    let ptr = &challenge as *const F as *const u64;
                    [*ptr, *ptr.add(1), *ptr.add(2)]
                };
                let w = extract_nonresidue_ext3::<F, Backend>();

                let n_f = f.len() * d;
                let f_buf: &mut [u64] =
                    unsafe { core::slice::from_raw_parts_mut(f.as_mut_ptr() as *mut u64, n_f) };
                crate::simd_sumcheck::reduce::ext3_reduce_in_place::<Backend>(f_buf, chg_raw, w);

                let n_g = g.len() * d;
                let g_buf: &mut [u64] =
                    unsafe { core::slice::from_raw_parts_mut(g.as_mut_ptr() as *mut u64, n_g) };
                crate::simd_sumcheck::reduce::ext3_reduce_in_place::<Backend>(g_buf, chg_raw, w);

                f.truncate(f.len() / 2);
                g.truncate(g.len() / 2);
                return Some(true);
            }
        }
    }

    None
}

// ─── Product evaluate ───────────────────────────────────────────────────────

/// Pairwise product sum: computes coefficients `(a, b)` of the degree-2
/// round polynomial from two evaluation vectors.
///
/// - `a = Σ f[2i] * g[2i]`  (even-even products)
/// - `b = Σ (f[2i] * g[2i+1] + f[2i+1] * g[2i])`  (cross-term)
///
/// SIMD-accelerated for Goldilocks base field.
pub fn pairwise_product_sum<F: Field>(f: &[F], g: &[F]) -> (F, F) {
    debug_assert_eq!(f.len(), g.len());

    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    if let Some(result) = try_simd_product_sum(f, g) {
        return result;
    }

    // Generic fallback
    crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate_slices(f, g)
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512ifma")
))]
fn try_simd_product_sum<F: Field>(f: &[F], g: &[F]) -> Option<(F, F)> {
    use crate::simd_sumcheck::dispatch::is_goldilocks_pub;

    #[cfg(target_arch = "aarch64")]
    type Backend = crate::simd_fields::goldilocks::neon::GoldilocksNeon;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    type Backend = crate::simd_fields::goldilocks::avx512::GoldilocksAvx512;

    if is_goldilocks_pub::<F>() {
        let f_raw: &[u64] =
            unsafe { core::slice::from_raw_parts(f.as_ptr() as *const u64, f.len()) };
        let g_raw: &[u64] =
            unsafe { core::slice::from_raw_parts(g.as_ptr() as *const u64, g.len()) };
        let (a, b) =
            crate::simd_sumcheck::evaluate::product_evaluate_parallel::<Backend>(f_raw, g_raw);

        use crate::simd_sumcheck::dispatch::u64_to_field_pub;
        return Some((u64_to_field_pub(a), u64_to_field_pub(b)));
    }

    // Ext2/ext3 path: AVX-512 only for now. NEON regresses on ext3 product
    // because it has no true vector 64×64 multiply — the SIMD wrapper adds
    // overhead without compute gain. Re-enable for NEON once a scalar-direct
    // path exists.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    {
        use crate::simd_sumcheck::dispatch::is_goldilocks_based_pub;
        if is_goldilocks_based_pub::<F>()
            && core::mem::size_of::<F>()
                == (F::extension_degree() as usize) * core::mem::size_of::<u64>()
        {
            let d = F::extension_degree() as usize;
            if d == 2 {
                return Some(simd_ext2_product_sum::<F, Backend>(f, g));
            } else if d == 3 {
                return Some(simd_ext3_product_sum::<F, Backend>(f, g));
            }
        }
    }

    None
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
const EXT_PRODUCT_CHUNK: usize = 1 << 14; // pairs per rayon chunk

/// Below this input size, `simd_ext{2,3}_product_sum` skips rayon entirely
/// and runs sequentially. Rayon's fork/join overhead dominates actual SIMD
/// compute for small inputs (profiling showed ~70% of short-call samples
/// in `_lll_lock_wake_private` / `mprotect`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
const EXT_PRODUCT_PARALLEL_THRESHOLD: usize = 1 << 17;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
fn aos_to_soa_ext2_serial(src: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let n = src.len() / 2;
    let mut c0 = vec![0u64; n];
    let mut c1 = vec![0u64; n];
    for i in 0..n {
        c0[i] = src[2 * i];
        c1[i] = src[2 * i + 1];
    }
    (c0, c1)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
fn aos_to_soa_ext3_serial(src: &[u64]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = src.len() / 3;
    let mut c0 = vec![0u64; n];
    let mut c1 = vec![0u64; n];
    let mut c2 = vec![0u64; n];
    for i in 0..n {
        c0[i] = src[3 * i];
        c1[i] = src[3 * i + 1];
        c2[i] = src[3 * i + 2];
    }
    (c0, c1, c2)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
fn aos_to_soa_ext2_par(src: &[u64]) -> (Vec<u64>, Vec<u64>) {
    use rayon::prelude::*;
    let n = src.len() / 2;
    let mut c0 = vec![0u64; n];
    let mut c1 = vec![0u64; n];
    let chunk = EXT_PRODUCT_CHUNK;
    c0.par_chunks_mut(chunk)
        .zip(c1.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(i, (c0_chunk, c1_chunk))| {
            let start = i * chunk;
            for j in 0..c0_chunk.len() {
                c0_chunk[j] = src[2 * (start + j)];
                c1_chunk[j] = src[2 * (start + j) + 1];
            }
        });
    (c0, c1)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
fn aos_to_soa_ext3_par(src: &[u64]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    use rayon::prelude::*;
    let n = src.len() / 3;
    let mut c0 = vec![0u64; n];
    let mut c1 = vec![0u64; n];
    let mut c2 = vec![0u64; n];
    let chunk = EXT_PRODUCT_CHUNK;
    c0.par_chunks_mut(chunk)
        .zip(c1.par_chunks_mut(chunk))
        .zip(c2.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(i, ((c0_chunk, c1_chunk), c2_chunk))| {
            let start = i * chunk;
            for j in 0..c0_chunk.len() {
                c0_chunk[j] = src[3 * (start + j)];
                c1_chunk[j] = src[3 * (start + j) + 1];
                c2_chunk[j] = src[3 * (start + j) + 2];
            }
        });
    (c0, c1, c2)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
fn simd_ext2_product_sum<F: Field, B: crate::simd_fields::SimdBaseField<Scalar = u64>>(
    f: &[F],
    g: &[F],
) -> (F, F) {
    use crate::simd_sumcheck::dispatch::extract_nonresidue_ext2;
    use rayon::prelude::*;

    let n = f.len();
    let f_raw: &[u64] = unsafe { core::slice::from_raw_parts(f.as_ptr() as *const u64, n * 2) };
    let g_raw: &[u64] = unsafe { core::slice::from_raw_parts(g.as_ptr() as *const u64, n * 2) };
    let w = extract_nonresidue_ext2::<F, B>();

    // Serial path for small inputs: rayon's fork/join cost would dominate.
    if n <= EXT_PRODUCT_PARALLEL_THRESHOLD {
        let (f_c0, f_c1) = aos_to_soa_ext2_serial(f_raw);
        let (g_c0, g_c1) = aos_to_soa_ext2_serial(g_raw);
        let (a, b) = crate::simd_sumcheck::reduce::ext2_soa_product_evaluate::<B>(
            &f_c0, &f_c1, &g_c0, &g_c1, w,
        );
        return (
            pack_ext_u64_to_field::<F>(&a),
            pack_ext_u64_to_field::<F>(&b),
        );
    }

    // Parallel AoS → SoA; one pass each for f and g.
    let ((f_c0, f_c1), (g_c0, g_c1)) =
        rayon::join(|| aos_to_soa_ext2_par(f_raw), || aos_to_soa_ext2_par(g_raw));

    // Chunks must be pair-aligned (even length). The last chunk may be odd
    // if n is odd, but pairwise_product_sum always receives even n (pairs).
    let chunk = EXT_PRODUCT_CHUNK;
    let (a, b) = f_c0
        .par_chunks(chunk)
        .zip(f_c1.par_chunks(chunk))
        .zip(g_c0.par_chunks(chunk))
        .zip(g_c1.par_chunks(chunk))
        .map(|(((fc0, fc1), gc0), gc1)| {
            crate::simd_sumcheck::reduce::ext2_soa_product_evaluate::<B>(fc0, fc1, gc0, gc1, w)
        })
        .reduce(
            || ([0u64; 2], [0u64; 2]),
            |(a1, b1), (a2, b2)| {
                (
                    [B::scalar_add(a1[0], a2[0]), B::scalar_add(a1[1], a2[1])],
                    [B::scalar_add(b1[0], b2[0]), B::scalar_add(b1[1], b2[1])],
                )
            },
        );

    (
        pack_ext_u64_to_field::<F>(&a),
        pack_ext_u64_to_field::<F>(&b),
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
fn simd_ext3_product_sum<F: Field, B: crate::simd_fields::SimdBaseField<Scalar = u64>>(
    f: &[F],
    g: &[F],
) -> (F, F) {
    use crate::simd_sumcheck::dispatch::extract_nonresidue_ext3;
    use rayon::prelude::*;

    let n = f.len();
    let f_raw: &[u64] = unsafe { core::slice::from_raw_parts(f.as_ptr() as *const u64, n * 3) };
    let g_raw: &[u64] = unsafe { core::slice::from_raw_parts(g.as_ptr() as *const u64, n * 3) };
    let w = extract_nonresidue_ext3::<F, B>();

    // Serial path for small inputs: rayon's fork/join cost would dominate.
    if n <= EXT_PRODUCT_PARALLEL_THRESHOLD {
        let (f_c0, f_c1, f_c2) = aos_to_soa_ext3_serial(f_raw);
        let (g_c0, g_c1, g_c2) = aos_to_soa_ext3_serial(g_raw);
        let (a, b) = crate::simd_sumcheck::reduce::ext3_soa_product_evaluate::<B>(
            &f_c0, &f_c1, &f_c2, &g_c0, &g_c1, &g_c2, w,
        );
        return (
            pack_ext_u64_to_field::<F>(&a),
            pack_ext_u64_to_field::<F>(&b),
        );
    }

    let ((f_c0, f_c1, f_c2), (g_c0, g_c1, g_c2)) =
        rayon::join(|| aos_to_soa_ext3_par(f_raw), || aos_to_soa_ext3_par(g_raw));

    let chunk = EXT_PRODUCT_CHUNK;
    let (a, b) = f_c0
        .par_chunks(chunk)
        .zip(f_c1.par_chunks(chunk))
        .zip(f_c2.par_chunks(chunk))
        .zip(g_c0.par_chunks(chunk))
        .zip(g_c1.par_chunks(chunk))
        .zip(g_c2.par_chunks(chunk))
        .map(|(((((fc0, fc1), fc2), gc0), gc1), gc2)| {
            crate::simd_sumcheck::reduce::ext3_soa_product_evaluate::<B>(
                fc0, fc1, fc2, gc0, gc1, gc2, w,
            )
        })
        .reduce(
            || ([0u64; 3], [0u64; 3]),
            |(a1, b1), (a2, b2)| {
                (
                    [
                        B::scalar_add(a1[0], a2[0]),
                        B::scalar_add(a1[1], a2[1]),
                        B::scalar_add(a1[2], a2[2]),
                    ],
                    [
                        B::scalar_add(b1[0], b2[0]),
                        B::scalar_add(b1[1], b2[1]),
                        B::scalar_add(b1[2], b2[2]),
                    ],
                )
            },
        );

    (
        pack_ext_u64_to_field::<F>(&a),
        pack_ext_u64_to_field::<F>(&b),
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
#[inline]
fn pack_ext_u64_to_field<F: Field>(limbs: &[u64]) -> F {
    debug_assert_eq!(
        core::mem::size_of::<F>(),
        limbs.len() * core::mem::size_of::<u64>()
    );
    unsafe {
        let mut out = core::mem::MaybeUninit::<F>::uninit();
        core::ptr::copy_nonoverlapping(limbs.as_ptr(), out.as_mut_ptr() as *mut u64, limbs.len());
        out.assume_init()
    }
}

// ─── Inner product ──────────────────────────────────────────────────────────

/// Dot product: `Σ f[i] * g[i]`.
///
/// SIMD-accelerated for Goldilocks base field.
pub fn inner_product<F: Field>(f: &[F], g: &[F]) -> F {
    debug_assert_eq!(f.len(), g.len());
    f.iter().zip(g.iter()).map(|(a, b)| *a * *b).sum()
    // Note: SIMD inner product would require extension multiply for ext fields.
    // For base field, the generic .sum() with rayon is already fast.
    // Future: add SIMD dispatch here.
}

// ─── Cross-field reduce ─────────────────────────────────────────────────────

/// Fold base-field evaluations with an extension-field challenge.
///
/// Each pair `(a, b)` in `data` (base field) is folded to
/// `EF::from(a) + challenge * (EF::from(b) - EF::from(a))` in the extension field.
///
/// Returns a new `Vec<EF>`.
pub fn cross_field_fold<BF: Field, EF: Field + From<BF>>(data: &[BF], challenge: EF) -> Vec<EF> {
    crate::multilinear::reductions::pairwise::cross_field_reduce(data, challenge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{F64Ext2, F64Ext3, F64};
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_pairwise_sum_base() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let data: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let (even, odd) = pairwise_sum(&data);

        let expected_even: F64 = data.iter().step_by(2).copied().sum();
        let expected_odd: F64 = data.iter().skip(1).step_by(2).copied().sum();

        assert_eq!(even, expected_even);
        assert_eq!(odd, expected_odd);
    }

    #[test]
    fn test_pairwise_sum_ext2() {
        let mut rng = test_rng();
        let n = 1 << 8;
        let data: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let (even, odd) = pairwise_sum(&data);

        let expected_even: F64Ext2 = data.iter().step_by(2).copied().sum();
        let expected_odd: F64Ext2 = data.iter().skip(1).step_by(2).copied().sum();

        assert_eq!(even, expected_even);
        assert_eq!(odd, expected_odd);
    }

    #[test]
    fn test_pairwise_sum_ext3() {
        let mut rng = test_rng();
        let n = 1 << 8;
        let data: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();

        let (even, odd) = pairwise_sum(&data);

        let expected_even: F64Ext3 = data.iter().step_by(2).copied().sum();
        let expected_odd: F64Ext3 = data.iter().skip(1).step_by(2).copied().sum();

        assert_eq!(even, expected_even);
        assert_eq!(odd, expected_odd);
    }

    #[test]
    fn test_fold_base() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let data: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let challenge = F64::rand(&mut rng);

        // Reference: manual fold
        let expected: Vec<F64> = data
            .chunks(2)
            .map(|c| c[0] + challenge * (c[1] - c[0]))
            .collect();

        let mut result = data;
        fold(&mut result, challenge);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold_ext2() {
        let mut rng = test_rng();
        let n = 1 << 8;
        let data: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let challenge = F64Ext2::rand(&mut rng);

        let expected: Vec<F64Ext2> = data
            .chunks(2)
            .map(|c| c[0] + challenge * (c[1] - c[0]))
            .collect();

        let mut result = data;
        fold(&mut result, challenge);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold_both_matches_separate() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let challenge = F64::rand(&mut rng);

        // Separate fold
        let mut f_sep = f.clone();
        let mut g_sep = g.clone();
        fold(&mut f_sep, challenge);
        fold(&mut g_sep, challenge);

        // Combined fold
        let mut f_both = f;
        let mut g_both = g;
        fold_both(&mut f_both, &mut g_both, challenge);

        assert_eq!(f_sep, f_both);
        assert_eq!(g_sep, g_both);
    }

    #[test]
    fn test_pairwise_product_sum_base() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let (a, b) = pairwise_product_sum(&f, &g);

        // Reference
        let expected_a: F64 = (0..n / 2).map(|k| f[2 * k] * g[2 * k]).sum();
        let expected_b: F64 = (0..n / 2)
            .map(|k| f[2 * k] * g[2 * k + 1] + f[2 * k + 1] * g[2 * k])
            .sum();

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_pairwise_product_sum_ext2() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let (a, b) = pairwise_product_sum(&f, &g);

        let expected_a: F64Ext2 = (0..n / 2).map(|k| f[2 * k] * g[2 * k]).sum();
        let expected_b: F64Ext2 = (0..n / 2)
            .map(|k| f[2 * k] * g[2 * k + 1] + f[2 * k + 1] * g[2 * k])
            .sum();

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }

    #[test]
    fn test_pairwise_product_sum_ext3() {
        let mut rng = test_rng();
        let n = 1 << 10;
        let f: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
        let g: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();

        let (a, b) = pairwise_product_sum(&f, &g);

        let expected_a: F64Ext3 = (0..n / 2).map(|k| f[2 * k] * g[2 * k]).sum();
        let expected_b: F64Ext3 = (0..n / 2)
            .map(|k| f[2 * k] * g[2 * k + 1] + f[2 * k + 1] * g[2 * k])
            .sum();

        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
    }
}
