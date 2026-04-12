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
    assert_eq!(f.len(), g.len());
    assert!(f.len().count_ones() == 1);

    // ── SIMD auto-dispatch ──
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx512ifma")
    ))]
    if let Some(result) =
        crate::simd_sumcheck::dispatch::try_simd_product_dispatch::<BF, EF>(f, g, transcript)
    {
        return result;
    }

    let num_rounds = f.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(EF, EF)> = vec![];
    let mut verifier_messages: Vec<EF> = vec![];

    // ── Round 0: evaluate in BF, lift to EF, cross-field reduce ──
    if num_rounds > 0 {
        // Use simd_ops for round 0 evaluate (SIMD-accelerated for Goldilocks)
        let msg_bf = crate::simd_ops::pairwise_product_sum(f, g);
        let msg = (EF::from(msg_bf.0), EF::from(msg_bf.1));

        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);

        let chg = transcript.read();
        verifier_messages.push(chg);

        // Cross-field reduce: BF evaluations + EF challenge → Vec<EF>
        let mut ef_f = crate::simd_ops::cross_field_fold(f, chg);
        let mut ef_g = crate::simd_ops::cross_field_fold(g, chg);

        // Remaining rounds work in EF.
        for _ in 1..num_rounds {
            // SIMD-accelerated product evaluate (dispatches for Goldilocks base)
            let msg = crate::simd_ops::pairwise_product_sum(&ef_f, &ef_g);

            prover_messages.push(msg);
            transcript.write(msg.0);
            transcript.write(msg.1);

            let chg = transcript.read();
            verifier_messages.push(chg);

            // SIMD-accelerated fold (dispatches for Goldilocks base + extensions)
            crate::simd_ops::fold(&mut ef_f, chg);
            crate::simd_ops::fold(&mut ef_g, chg);
        }
    }

    ProductSumcheck {
        verifier_messages,
        prover_messages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
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
}
