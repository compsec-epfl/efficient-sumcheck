//! Inner product sumcheck protocol.
//!
//! Given two evaluation vectors `f` and `g` representing multilinear polynomials on
//! the boolean hypercube `{0,1}^n`, the [`inner_product_sumcheck`] function executes
//! `n` rounds of the product sumcheck protocol computing `∑_x f(x)·g(x)`, and returns
//! the resulting [`ProductSumcheck`] transcript.
//!
//! The function is parameterized by two field types:
//! - `BF` (base field): the field the evaluations live in
//! - `EF` (extension field): the field challenges are sampled from
//!
//! When no extension field is needed, set `EF = BF`.
//!
//! # Example
//!
//! ```text
//! use efficient_sumcheck::{inner_product_sumcheck, ProductSumcheck};
//! use efficient_sumcheck::transcript::SanityTranscript;
//!
//! // No extension field (BF = EF):
//! let mut f = vec![F::from(1), F::from(2), F::from(3), F::from(4)];
//! let mut g = vec![F::from(5), F::from(6), F::from(7), F::from(8)];
//! let mut transcript = SanityTranscript::new(&mut rng);
//! let result: ProductSumcheck<F> = inner_product_sumcheck(&mut f, &mut g, &mut transcript);
//! ```

use ark_std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;

use ark_ff::Field;

use crate::transcript::Transcript;

pub use crate::multilinear_product::ProductSumcheck;

pub type FastMap<V> = HashMap<usize, V, BuildNoHashHasher<usize>>;

pub fn batched_constraint_poly<F: Field>(
    dense_polys: &Vec<Vec<F>>,
    sparse_polys: &FastMap<F>,
) -> Vec<F> {
    fn sum_columns<F: Field>(matrix: &Vec<Vec<F>>) -> Vec<F> {
        if matrix.is_empty() {
            return vec![];
        }
        let mut result = vec![F::ZERO; matrix[0].len()];
        for row in matrix {
            for (i, &val) in row.iter().enumerate() {
                result[i] += val;
            }
        }
        result
    }
    let mut res = sum_columns(dense_polys);
    for (k, v) in sparse_polys.iter() {
        res[*k] += v;
    }
    res
}

// [CBBZ23] hyperplonk optimization
/// Accumulate eq polynomial evaluations at binary query points into a sparse map.
/// Skips indices `0..=s`.
pub fn accumulate_sparse_evaluations<F: Field>(
    zetas: Vec<&[F]>,
    eq_evals: Vec<F>,
    s: usize,
    r: usize,
) -> FastMap<F> {
    let mut result = FastMap::default();
    for i in 1 + s..r {
        let index = zetas[i]
            .iter()
            .enumerate()
            .filter_map(|(j, bit)| bit.is_one().then_some(1 << j))
            .sum::<usize>();
        *result.entry(index).or_insert(F::zero()) += &eq_evals[i];
    }
    result
}

/// Run the inner product sumcheck protocol over two evaluation vectors,
/// using a generic [`Transcript`] for Fiat-Shamir (or sanity/random challenges).
///
/// `BF` is the base field of the evaluations, `EF` is the extension field for challenges.
/// When `BF = EF`, this is the standard single-field inner product sumcheck.
/// When `BF ≠ EF`, round 0 evaluates in `BF` and lifts to `EF`, then subsequent
/// rounds work entirely in `EF`.
///
/// Each round:
/// 1. Computes `(a, b)` — the constant and linear coefficients of the degree-2
///    round polynomial `q(x) = a + bx + cx²`.
/// 2. Writes them to the transcript (2 field elements).
/// 3. Reads the verifier's challenge from the transcript (1 field element).
/// 4. Reduces both evaluation vectors by folding with the challenge.
///
/// The verifier derives `c = claim - 2a - b` from the constraint `q(0) + q(1) = claim`.
pub fn inner_product_sumcheck<BF: Field, EF: Field + From<BF>>(
    f: &mut [BF],
    g: &mut [BF],
    transcript: &mut impl Transcript<EF>,
) -> ProductSumcheck<EF> {
    inner_product_sumcheck_with_hook(f, g, transcript, |_, _| {})
}

/// Like [`inner_product_sumcheck`], but calls `hook(round_idx, transcript)`
/// each round *after* the prover message is written and *before* the verifier
/// challenge is read.
///
/// See [`crate::multilinear_sumcheck_with_hook`] for the motivating use case
/// (per-round proof-of-work grinding, etc.).
/// Partial inner-product sumcheck: runs `max_rounds` rounds and stops.
///
/// Folds `f` and `g` in place (truncating them to length `original / 2^max_rounds`)
/// so the caller can feed them into a subsequent partial sumcheck call. This
/// is the shape recursive IOPs (e.g. whir) need: between rounds the caller
/// commits, opens, and mutates the running claim before continuing.
///
/// Requires `BF = EF = F` (no cross-field lift). Uses SIMD-accelerated
/// [`crate::simd_ops::pairwise_product_sum`] and [`crate::simd_ops::fold_both`]
/// per round, so SIMD dispatch happens under the hood — but without the
/// fused reduce+evaluate optimization the full-sumcheck dispatch has. For
/// whir-style calls where `max_rounds` is small (e.g. a folding factor), this
/// is the right tradeoff.
///
/// `ProductSumcheck::final_evaluations` is populated only if `max_rounds`
/// reduces `f` to length 1 (i.e., a complete sumcheck); otherwise
/// `(F::ZERO, F::ZERO)`. The caller uses `f[0]` / `g[0]` of the returned
/// folded vectors for the intermediate state.
pub fn inner_product_sumcheck_partial_with_hook<F, T, H>(
    f: &mut Vec<F>,
    g: &mut Vec<F>,
    transcript: &mut T,
    max_rounds: usize,
    mut hook: H,
) -> ProductSumcheck<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
{
    assert_eq!(f.len(), g.len());
    assert!(f.len().count_ones() == 1, "length must be a power of 2");
    let total_rounds = f.len().trailing_zeros() as usize;
    assert!(
        max_rounds <= total_rounds,
        "max_rounds ({max_rounds}) exceeds available rounds ({total_rounds})"
    );

    // Fast path: SoA-persistent SIMD dispatch for Goldilocks ext2/ext3 on
    // AVX-512. Keeps SoA state across all `max_rounds` rounds — one
    // AoS→SoA conversion at entry, one SoA→AoS at exit (vs the per-round
    // round-trip of the fallback loop).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512ifma"))]
    {
        if let Some(result) = crate::simd_sumcheck::dispatch::try_simd_ext_product_partial_dispatch(
            f, g, transcript, max_rounds, &mut hook,
        ) {
            return result;
        }
    }

    let mut prover_messages: Vec<(F, F)> = Vec::with_capacity(max_rounds);
    let mut verifier_messages: Vec<F> = Vec::with_capacity(max_rounds);

    for round in 0..max_rounds {
        let msg = crate::simd_ops::pairwise_product_sum(f, g);

        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        hook(round, transcript);

        let chg = transcript.read();
        verifier_messages.push(chg);

        crate::simd_ops::fold_both(f, g, chg);
    }

    let final_evaluations = if f.len() == 1 {
        (f[0], g[0])
    } else {
        (F::ZERO, F::ZERO)
    };

    ProductSumcheck {
        prover_messages,
        verifier_messages,
        final_evaluations,
    }
}

pub fn inner_product_sumcheck_with_hook<BF, EF, T, H>(
    f: &mut [BF],
    g: &mut [BF],
    transcript: &mut T,
    mut hook: H,
) -> ProductSumcheck<EF>
where
    BF: Field,
    EF: Field + From<BF>,
    T: Transcript<EF>,
    H: FnMut(usize, &mut T),
{
    assert_eq!(f.len(), g.len());
    assert!(f.len().count_ones() == 1);

    // ── SIMD auto-dispatch ──
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    {
        // Try base-field dispatch first (BF == EF == Goldilocks base)
        if let Some(result) =
            crate::simd_sumcheck::dispatch::try_simd_product_dispatch::<BF, EF, T, H>(
                f, g, transcript, &mut hook,
            )
        {
            return result;
        }
        // Try extension-field dispatch (BF == EF == Goldilocks ext2)
        if let Some(result) =
            crate::simd_sumcheck::dispatch::try_simd_ext_product_dispatch::<BF, EF, T, H>(
                f, g, transcript, &mut hook,
            )
        {
            return result;
        }
    }

    let num_rounds = f.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = vec![];
    let mut verifier_messages: Vec<EF> = vec![];
    let mut final_evaluations = (EF::ZERO, EF::ZERO);

    // ── Round 0: evaluate in BF, lift to EF, cross-field reduce ──
    if num_rounds > 0 {
        // Use simd_ops for round 0 evaluate (SIMD-accelerated for Goldilocks)
        let msg_bf = crate::simd_ops::pairwise_product_sum(f, g);
        let msg = (EF::from(msg_bf.0), EF::from(msg_bf.1));

        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        hook(0, transcript);

        let chg = transcript.read();
        verifier_messages.push(chg);

        // Cross-field reduce: BF evaluations + EF challenge → Vec<EF>
        let mut ef_f = crate::simd_ops::cross_field_fold(f, chg);
        let mut ef_g = crate::simd_ops::cross_field_fold(g, chg);

        // Remaining rounds work in EF.
        for round in 1..num_rounds {
            // SIMD-accelerated product evaluate (dispatches for Goldilocks base)
            let msg = crate::simd_ops::pairwise_product_sum(&ef_f, &ef_g);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            hook(round, transcript);

            let chg = transcript.read();
            verifier_messages.push(chg);

            // SIMD-accelerated fold (dispatches for Goldilocks base + extensions)
            crate::simd_ops::fold(&mut ef_f, chg);
            crate::simd_ops::fold(&mut ef_g, chg);
        }

        debug_assert_eq!(ef_f.len(), 1);
        debug_assert_eq!(ef_g.len(), 1);
        final_evaluations = (ef_f[0], ef_g[0]);
    }

    ProductSumcheck {
        verifier_messages,
        prover_messages,
        final_evaluations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{AdditiveGroup, UniformRand};
    use ark_std::test_rng;

    use crate::tests::F64;

    const NUM_VARS: usize = 4; // vectors of length 2^4 = 16

    #[test]
    fn test_inner_product_sumcheck_sanity() {
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut transcript = SanityTranscript::new(&mut rng);
        let result = inner_product_sumcheck::<F64, F64>(&mut f, &mut g, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }

    #[test]
    fn test_simd_parity_with_generic() {
        // Compare SIMD auto-dispatch path against the generic TimeProductProver path.
        // Both should produce identical prover messages given the same transcript.
        use crate::transcript::SanityTranscript;

        let mut eval_rng = test_rng();
        let n = 1usize << 8;
        let f_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut eval_rng)).collect();
        let g_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut eval_rng)).collect();

        // Run via inner_product_sumcheck (SIMD dispatched for F64/Goldilocks)
        let mut rng1 = test_rng();
        let mut f1 = f_orig.clone();
        let mut g1 = g_orig.clone();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let simd_result = inner_product_sumcheck::<F64, F64>(&mut f1, &mut g1, &mut t1);

        // Run the generic path manually (bypass SIMD dispatch)
        let mut rng2 = test_rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let num_rounds = n.trailing_zeros() as usize;
        let mut generic_prover_msgs = Vec::with_capacity(num_rounds);
        let mut generic_verifier_msgs = Vec::with_capacity(num_rounds);

        use crate::multilinear::reductions::pairwise;
        use crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;

        // Round 0
        let msg = pairwise_product_evaluate(&[f_orig.clone(), g_orig.clone()]);
        generic_prover_msgs.push(msg);
        t2.write(msg.0);
        t2.write(msg.1);
        let chg: F64 = t2.read();
        generic_verifier_msgs.push(chg);
        let mut ef_f = pairwise::cross_field_reduce(&f_orig, chg);
        let mut ef_g = pairwise::cross_field_reduce(&g_orig, chg);

        // Rounds 1+
        for _ in 1..num_rounds {
            let msg = pairwise_product_evaluate(&[ef_f.clone(), ef_g.clone()]);
            generic_prover_msgs.push(msg);
            t2.write(msg.0);
            t2.write(msg.1);
            let chg: F64 = t2.read();
            generic_verifier_msgs.push(chg);
            pairwise::reduce_evaluations(&mut ef_f, chg);
            pairwise::reduce_evaluations(&mut ef_g, chg);
        }

        // Compare
        assert_eq!(simd_result.prover_messages.len(), generic_prover_msgs.len());
        for (i, (s, g)) in simd_result
            .prover_messages
            .iter()
            .zip(generic_prover_msgs.iter())
            .enumerate()
        {
            assert_eq!(s.0, g.0, "a mismatch at round {i}");
            assert_eq!(s.1, g.1, "b mismatch at round {i}");
        }
        assert_eq!(simd_result.verifier_messages, generic_verifier_msgs);
    }

    #[test]
    fn test_inner_product_sumcheck_spongefish() {
        use crate::transcript::SpongefishTranscript;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let domsep = spongefish::domain_separator!("test-inner-product-sumcheck"; module_path!())
            .instance(b"test");

        let prover_state = domsep.std_prover();
        let mut transcript = SpongefishTranscript::new(prover_state);
        let result = inner_product_sumcheck::<F64, F64>(&mut f, &mut g, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }

    #[test]
    fn test_inner_product_extension_field() {
        // Test inner product sumcheck with BF = EF = F64Ext2.
        use crate::tests::F64Ext2;
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();
        let n = 1 << 6;
        let mut f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let mut g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let mut transcript = SanityTranscript::new(&mut rng);
        let result = inner_product_sumcheck::<F64Ext2, F64Ext2>(&mut f, &mut g, &mut transcript);

        assert_eq!(result.prover_messages.len(), 6);
        assert_eq!(result.verifier_messages.len(), 6);
    }

    /// Sanity check for the ext2 IP SIMD dispatch path at a small size (below the
    /// parallel threshold). Pre-existing test_inner_product_extension_field only
    /// checks message counts, so this catches round-0 evaluate mismatches too.
    #[test]
    fn test_ip_ext2_small_matches_reference() {
        use crate::multilinear::reductions::pairwise;
        use crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;
        use crate::tests::F64Ext2;
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();
        let n: usize = 1 << 8;
        let f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let mut rng1 = test_rng();
        let mut f1 = f.clone();
        let mut g1 = g.clone();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let simd_result =
            inner_product_sumcheck::<F64Ext2, F64Ext2>(&mut f1, &mut g1, &mut t1);

        let mut rng2 = test_rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let num_rounds = n.trailing_zeros() as usize;
        let mut ref_msgs = Vec::with_capacity(num_rounds);
        let mut ef_f = f;
        let mut ef_g = g;
        for _ in 0..num_rounds {
            let msg = pairwise_product_evaluate(&[ef_f.clone(), ef_g.clone()]);
            ref_msgs.push(msg);
            t2.write(msg.0);
            t2.write(msg.1);
            let chg: F64Ext2 = t2.read();
            pairwise::reduce_evaluations(&mut ef_f, chg);
            pairwise::reduce_evaluations(&mut ef_g, chg);
        }

        for (i, (s, r)) in simd_result
            .prover_messages
            .iter()
            .zip(ref_msgs.iter())
            .enumerate()
        {
            assert_eq!(s.0, r.0, "a mismatch at round {i}");
            assert_eq!(s.1, r.1, "b mismatch at round {i}");
        }
    }

    /// Exercises the rayon-parallel SoA product reduce path (n > 2^17 threshold).
    #[test]
    fn test_ip_ext2_parallel_path_matches_reference() {
        use crate::multilinear::reductions::pairwise;
        use crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;
        use crate::tests::F64Ext2;
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();
        let n: usize = 1 << 18;
        let f: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let g: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        // SIMD path (hits parallel dispatch above threshold)
        let mut rng1 = test_rng();
        let mut f1 = f.clone();
        let mut g1 = g.clone();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let parallel_result =
            inner_product_sumcheck::<F64Ext2, F64Ext2>(&mut f1, &mut g1, &mut t1);

        // Reference: generic pairwise evaluate+reduce loop
        let mut rng2 = test_rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let num_rounds = n.trailing_zeros() as usize;
        let mut ref_msgs = Vec::with_capacity(num_rounds);
        let mut ef_f = f;
        let mut ef_g = g;
        for _ in 0..num_rounds {
            let msg = pairwise_product_evaluate(&[ef_f.clone(), ef_g.clone()]);
            ref_msgs.push(msg);
            t2.write(msg.0);
            t2.write(msg.1);
            let chg: F64Ext2 = t2.read();
            pairwise::reduce_evaluations(&mut ef_f, chg);
            pairwise::reduce_evaluations(&mut ef_g, chg);
        }

        assert_eq!(parallel_result.prover_messages.len(), ref_msgs.len());
        for (i, (s, ref_msg)) in parallel_result
            .prover_messages
            .iter()
            .zip(ref_msgs.iter())
            .enumerate()
        {
            assert_eq!(s.0, ref_msg.0, "a mismatch at round {i}");
            assert_eq!(s.1, ref_msg.1, "b mismatch at round {i}");
        }
    }

    #[test]
    fn test_ip_ext3_parallel_path_matches_reference() {
        use crate::multilinear::reductions::pairwise;
        use crate::multilinear_product::provers::time::reductions::pairwise::pairwise_product_evaluate;
        use crate::tests::F64Ext3;
        use crate::transcript::SanityTranscript;

        let mut rng = test_rng();
        let n: usize = 1 << 18;
        let f: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();
        let g: Vec<F64Ext3> = (0..n).map(|_| F64Ext3::rand(&mut rng)).collect();

        let mut rng1 = test_rng();
        let mut f1 = f.clone();
        let mut g1 = g.clone();
        let mut t1 = SanityTranscript::new(&mut rng1);
        let parallel_result =
            inner_product_sumcheck::<F64Ext3, F64Ext3>(&mut f1, &mut g1, &mut t1);

        let mut rng2 = test_rng();
        let mut t2 = SanityTranscript::new(&mut rng2);
        let num_rounds = n.trailing_zeros() as usize;
        let mut ref_msgs = Vec::with_capacity(num_rounds);
        let mut ef_f = f;
        let mut ef_g = g;
        for _ in 0..num_rounds {
            let msg = pairwise_product_evaluate(&[ef_f.clone(), ef_g.clone()]);
            ref_msgs.push(msg);
            t2.write(msg.0);
            t2.write(msg.1);
            let chg: F64Ext3 = t2.read();
            pairwise::reduce_evaluations(&mut ef_f, chg);
            pairwise::reduce_evaluations(&mut ef_g, chg);
        }

        for (i, (s, ref_msg)) in parallel_result
            .prover_messages
            .iter()
            .zip(ref_msgs.iter())
            .enumerate()
        {
            assert_eq!(s.0, ref_msg.0, "a mismatch at round {i}");
            assert_eq!(s.1, ref_msg.1, "b mismatch at round {i}");
        }
    }

    fn fold_multilinear<F: ark_ff::Field>(evals: &[F], challenges: &[F]) -> F {
        let mut current = evals.to_vec();
        for &chg in challenges {
            let mut next = Vec::with_capacity(current.len() / 2);
            for pair in current.chunks(2) {
                next.push(pair[0] + chg * (pair[1] - pair[0]));
            }
            current = next;
        }
        debug_assert_eq!(current.len(), 1);
        current[0]
    }

    #[test]
    fn test_final_evaluations_match_independent_fold_base() {
        use crate::transcript::SanityTranscript;

        let num_vars = 8;
        let n = 1 << num_vars;
        let mut rng = test_rng();
        let f_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut f = f_orig.clone();
        let mut g = g_orig.clone();
        let mut transcript = SanityTranscript::new(&mut rng);
        let result = inner_product_sumcheck::<F64, F64>(&mut f, &mut g, &mut transcript);

        let expected_f = fold_multilinear(&f_orig, &result.verifier_messages);
        let expected_g = fold_multilinear(&g_orig, &result.verifier_messages);
        assert_eq!(result.final_evaluations.0, expected_f, "f final mismatch");
        assert_eq!(result.final_evaluations.1, expected_g, "g final mismatch");
    }

    #[test]
    fn test_final_evaluations_match_independent_fold_ext2() {
        use crate::tests::F64Ext2;
        use crate::transcript::SanityTranscript;

        let num_vars = 8;
        let n = 1 << num_vars;
        let mut rng = test_rng();
        let f_orig: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();
        let g_orig: Vec<F64Ext2> = (0..n).map(|_| F64Ext2::rand(&mut rng)).collect();

        let mut f = f_orig.clone();
        let mut g = g_orig.clone();
        let mut transcript = SanityTranscript::new(&mut rng);
        let result =
            inner_product_sumcheck::<F64Ext2, F64Ext2>(&mut f, &mut g, &mut transcript);

        let expected_f = fold_multilinear(&f_orig, &result.verifier_messages);
        let expected_g = fold_multilinear(&g_orig, &result.verifier_messages);
        assert_eq!(result.final_evaluations.0, expected_f, "ext2 f final mismatch");
        assert_eq!(result.final_evaluations.1, expected_g, "ext2 g final mismatch");
    }

    #[test]
    fn test_partial_split_matches_full() {
        // Running partial(N rounds) then partial(M rounds) on the folded state
        // must produce the same transcript as a single full run of N+M rounds.
        use crate::transcript::SanityTranscript;

        let num_vars = 8;
        let n = 1 << num_vars;
        let split_at = 3;
        let mut rng = test_rng();
        let f_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Full: single end-to-end run.
        let mut rng1 = test_rng();
        let mut f_full = f_orig.clone();
        let mut g_full = g_orig.clone();
        let mut t_full = SanityTranscript::new(&mut rng1);
        let full = inner_product_sumcheck::<F64, F64>(&mut f_full, &mut g_full, &mut t_full);

        // Split: two partial runs on the same transcript.
        let mut rng2 = test_rng();
        let mut f = f_orig.clone();
        let mut g = g_orig.clone();
        let mut t_split = SanityTranscript::new(&mut rng2);
        let first = inner_product_sumcheck_partial_with_hook(
            &mut f,
            &mut g,
            &mut t_split,
            split_at,
            |_, _| {},
        );
        let second = inner_product_sumcheck_partial_with_hook(
            &mut f,
            &mut g,
            &mut t_split,
            num_vars - split_at,
            |_, _| {},
        );

        let mut split_prover_msgs = first.prover_messages.clone();
        split_prover_msgs.extend(second.prover_messages.iter().copied());
        let mut split_verifier_msgs = first.verifier_messages.clone();
        split_verifier_msgs.extend(second.verifier_messages.iter().copied());

        assert_eq!(split_prover_msgs, full.prover_messages, "prover msgs");
        assert_eq!(split_verifier_msgs, full.verifier_messages, "verifier msgs");
        assert_eq!(second.final_evaluations, full.final_evaluations, "final");
        assert_eq!(first.final_evaluations, (F64::ZERO, F64::ZERO), "partial final should be zero");
    }

    #[test]
    fn test_with_hook_called_once_per_round() {
        use crate::transcript::SanityTranscript;
        use std::cell::RefCell;

        let num_vars = 6;
        let n = 1 << num_vars;
        let mut rng = test_rng();
        let mut f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut transcript = SanityTranscript::new(&mut rng);

        let calls = RefCell::new(Vec::<usize>::new());
        let result = inner_product_sumcheck_with_hook::<F64, F64, _, _>(
            &mut f,
            &mut g,
            &mut transcript,
            |round, _t| calls.borrow_mut().push(round),
        );

        assert_eq!(result.prover_messages.len(), num_vars);
        let calls = calls.into_inner();
        assert_eq!(calls, (0..num_vars).collect::<Vec<_>>(), "hook must be called once per round in order");
    }

    #[test]
    fn test_with_hook_injects_into_transcript() {
        use crate::transcript::SpongefishTranscript;

        let num_vars = 4;
        let n = 1 << num_vars;

        let mut rng = test_rng();
        let f_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let g_orig: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let run = |tag: F64, f: Vec<F64>, g: Vec<F64>| {
            let mut f = f;
            let mut g = g;
            let domsep = spongefish::domain_separator!("hook-test-ip"; module_path!())
                .instance(b"test");
            let prover_state = domsep.std_prover();
            let mut transcript = SpongefishTranscript::new(prover_state);
            inner_product_sumcheck_with_hook::<F64, F64, _, _>(
                &mut f,
                &mut g,
                &mut transcript,
                move |_round, t| {
                    t.write(tag);
                },
            )
        };

        let result_a = run(F64::from(1u64), f_orig.clone(), g_orig.clone());
        let result_b = run(F64::from(2u64), f_orig, g_orig);

        assert_ne!(
            result_a.verifier_messages[0],
            result_b.verifier_messages[0],
            "hook writes must affect Fiat-Shamir state"
        );
    }
}
