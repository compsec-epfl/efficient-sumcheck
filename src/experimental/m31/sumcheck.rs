use ark_ff::Field;

use crate::Sumcheck;
use crate::multilinear::pairwise;
use crate::tests::{Fp4SmallM31, SmallM31};
use crate::experimental::m31::{
    evaluate_bf::evaluate_bf, evaluate_ef::evaluate_ef, reduce_bf::reduce_bf, reduce_ef::reduce_ef,
};
use crate::{experimental::fiat_shamir::FiatShamir};

// pub struct Sumcheck<F: Field> {
//     prover_messages: Vec<(F, F)>,
//     verifier_messages: Vec<F>,
// }

pub fn prove(evals: &[SmallM31], fs: &mut impl FiatShamir<Fp4SmallM31>) -> Sumcheck<Fp4SmallM31> {
    let len = evals.len();
    assert!(len.count_ones() == 1, "evals len must be power of 2");
    let num_vars = len.trailing_zeros();
    let mut prover_messages = vec![];
    let mut verifier_messages = vec![];
    let mut new_evals = vec![];

    for i in 0..num_vars {
        if i == 0 {
            // evaluate
            let sums: (SmallM31, SmallM31) = evaluate_bf::<2_147_483_647>(evals);
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
            new_evals = reduce_bf(evals, verifier_message);
        } else {
            // evaluate
            let sums = if i < num_vars - 1 {
                evaluate_ef::<2_147_483_647>(&new_evals)
            } else {
                pairwise::evaluate(&new_evals)
            };

            prover_messages.push(sums);

            // absorb
            fs.absorb(sums.0);
            fs.absorb(sums.1);
            if i != num_vars - 1 {
                // squeeze
                let verifier_message = fs.squeeze();
                verifier_messages.push(verifier_message);
                // reduce
                reduce_ef(&mut new_evals, verifier_message);
            }
        }
    }
    Sumcheck::<Fp4SmallM31> {
        prover_messages,
        verifier_messages,
        is_accepted: true,
    }
}

#[cfg(test)]
mod tests {
    use super::Sumcheck;
    use crate::{
        multilinear::{ReduceMode, TimeProver},
        prover::Prover,
        tests::{BenchStream, Fp4SmallM31, SmallM31},
        experimental::{fiat_shamir::BenchFiatShamir, m31::sumcheck::prove},
    };
    use ark_ff::Field;

    #[test]
    fn sanity() {
        const NUM_VARIABLES: usize = 16;

        // take an evaluation stream
        let evaluation_stream: BenchStream<SmallM31> = BenchStream::new(NUM_VARIABLES);
        let claim = evaluation_stream.claimed_sum;

        // time_prover
        let mut time_prover =
            TimeProver::<SmallM31, BenchStream<SmallM31>>::new(<TimeProver<
                SmallM31,
                BenchStream<SmallM31>,
            > as Prover<SmallM31>>::ProverConfig::new(
                claim,
                NUM_VARIABLES,
                evaluation_stream.clone(),
                ReduceMode::Pairwise,
            ));
        let time_prover_transcript = Sumcheck::<SmallM31>::prove::<
            BenchStream<SmallM31>,
            TimeProver<SmallM31, BenchStream<SmallM31>>,
        >(&mut time_prover, &mut ark_std::test_rng());

        // take the same evaluation stream
        let len = 1 << NUM_VARIABLES;
        let evals: Vec<SmallM31> = (0..len).map(|x| SmallM31::from(x as u32)).collect();
        let mut fs = BenchFiatShamir::<Fp4SmallM31, _>::new(ark_std::test_rng());
        let transcript = prove(&evals, &mut fs);

        // were the challenges the same?
        let verifier_messages_fp4: Vec<Fp4SmallM31> = time_prover_transcript
            .verifier_messages
            .iter()
            .map(|a| Fp4SmallM31::from_base_prime_field(*a))
            .collect();
        assert_eq!(
            verifier_messages_fp4, transcript.verifier_messages,
            "challenges not same"
        );

        let prover_messages_fp4: Vec<(Fp4SmallM31, Fp4SmallM31)> = time_prover_transcript
            .prover_messages
            .iter()
            .map(|&(a, b)| {
                (
                    Fp4SmallM31::from_base_prime_field(a),
                    Fp4SmallM31::from_base_prime_field(b),
                )
            })
            .collect();
        assert_eq!(
            prover_messages_fp4, transcript.prover_messages,
            "prover messages not same"
        );
    }
}
