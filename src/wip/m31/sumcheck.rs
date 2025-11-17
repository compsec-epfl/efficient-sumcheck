use ark_ff::Field;

use crate::multilinear::pairwise;
use crate::tests::{Fp4SmallM31, SmallM31};
use crate::wip::m31::vectorized_reductions::pairwise::{evaluate_bf, reduce_evaluations_bf};
use crate::{wip::fiat_shamir::FiatShamir, Sumcheck};

fn prove(evals: &[SmallM31], fs: &mut impl FiatShamir<Fp4SmallM31>) -> Sumcheck<Fp4SmallM31> {
    let len = evals.len();
    assert!(len.count_ones() == 1, "evals len must be power of 2");
    let num_vars = len.trailing_zeros();
    let mut prover_messages = vec![];
    let mut verifier_messages = vec![];
    let mut new_evals = vec![];

    for i in 0..num_vars {
        if i == 0 {
            // evaluate
            let sums: (SmallM31, SmallM31) = evaluate_bf(evals);
            // promote to EF
            let (sum_0, sum_1) = (
                Fp4SmallM31::from_base_prime_field(sums.0),
                Fp4SmallM31::from_base_prime_field(sums.1),
            );
            prover_messages.push((sum_0, sum_1));
            // absorb
            fs.absorb(sum_0);
            fs.absorb(sum_1);
            // squeeze
            let verifier_message = fs.squeeze();
            verifier_messages.push(verifier_message);
            // reduce
            new_evals = reduce_evaluations_bf(evals, verifier_message);
        } else {
            // evaluate
            let sums = pairwise::evaluate(&new_evals);
            // absorb
            fs.absorb(sums.0);
            fs.absorb(sums.1);
            // squeeze
            let verifier_message = fs.squeeze();
            verifier_messages.push(verifier_message);
            // reduce
            pairwise::reduce_evaluations(&mut new_evals, verifier_message);
        }
    }
    Sumcheck::<Fp4SmallM31> {
        prover_messages,
        verifier_messages,
        is_accepted: true, // this should be removed
    }
}
