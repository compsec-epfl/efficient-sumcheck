use ark_ff::Field;

use crate::experimental::fiat_shamir::FiatShamir;
// Import the Goldilocks versions of your SIMD functions
use crate::experimental::goldilocks::{
    evaluate_bf::evaluate_bf, evaluate_ef::evaluate_ef, 
    reduce_bf::reduce_bf, reduce_ef::reduce_ef,
};
use crate::multilinear::pairwise;
// Swap types to Goldilocks
use crate::tests::{Fp2SmallGoldilocks, SmallGoldilocks};
use crate::Sumcheck;

pub fn prove(
    evals: &[SmallGoldilocks], 
    fs: &mut impl FiatShamir<Fp2SmallGoldilocks>
) -> Sumcheck<Fp2SmallGoldilocks> {
    let len = evals.len();
    assert!(len.count_ones() == 1, "evals len must be power of 2");
    let num_vars = len.trailing_zeros();
    let mut prover_messages = vec![];
    let mut verifier_messages = vec![];
    let mut new_evals = vec![];

    for i in 0..num_vars {
        if i == 0 {
            // 1. Evaluate Base Field (Goldilocks 2^64 - 2^32 + 1)
            // Use the Goldilocks modulus constant
            let sums: (SmallGoldilocks, SmallGoldilocks) = 
                evaluate_bf::<18446744069414584321>(evals);
            
            // 2. Promote to Extension Field (Fp2)
            let (sum_0, sum_1) = (
                Fp2SmallGoldilocks::from_base_prime_field(sums.0),
                Fp2SmallGoldilocks::from_base_prime_field(sums.1),
            );
            prover_messages.push((sum_0, sum_1));
            
            // 3. Fiat-Shamir
            fs.absorb(sum_0);
            fs.absorb(sum_1);
            let verifier_message = fs.squeeze();
            verifier_messages.push(verifier_message);
            
            // 4. Reduce Base Field to Extension Field
            // This reduction takes base field elements and a challenge in Fp2
            new_evals = reduce_bf(evals, verifier_message);
        } else {
            // Evaluate Extension Field
            let sums = if i < num_vars - 1 {
                evaluate_ef::<18446744069414584321>(&new_evals)
            } else {
                // Last step uses standard pairwise evaluation
                pairwise::evaluate(&new_evals)
            };

            prover_messages.push(sums);

            fs.absorb(sums.0);
            fs.absorb(sums.1);
            
            if i != num_vars - 1 {
                let verifier_message = fs.squeeze();
                verifier_messages.push(verifier_message);
                // Reduce Extension Field
                reduce_ef(&mut new_evals, verifier_message);
            }
        }
    }
    
    Sumcheck::<Fp2SmallGoldilocks> {
        prover_messages,
        verifier_messages,
        is_accepted: true,
    }
}


// has error
#[cfg(test)]
mod tests {
    use super::Sumcheck;
    use crate::{
        experimental::{fiat_shamir::BenchFiatShamir, goldilocks::sumcheck::prove},
        multilinear::{ReduceMode, TimeProver},
        prover::Prover,
        tests::{BenchStream, Fp2SmallGoldilocks, SmallGoldilocks},
    };
    use ark_ff::Field;

    #[test]
    fn sanity() {
        const NUM_VARIABLES: usize = 16;

        // take an evaluation stream
        let evaluation_stream: BenchStream<SmallGoldilocks> = BenchStream::new(NUM_VARIABLES);
        let claim = evaluation_stream.claimed_sum;

        // time_prover
        let mut time_prover =
            TimeProver::<SmallGoldilocks, BenchStream<SmallGoldilocks>>::new(<TimeProver<
                SmallGoldilocks,
                BenchStream<SmallGoldilocks>,
            > as Prover<SmallGoldilocks>>::ProverConfig::new(
                claim,
                NUM_VARIABLES,
                evaluation_stream.clone(),
                ReduceMode::Pairwise,
            ));
        let time_prover_transcript = Sumcheck::<SmallGoldilocks>::prove::<
            BenchStream<SmallGoldilocks>,
            TimeProver<SmallGoldilocks, BenchStream<SmallGoldilocks>>,
        >(&mut time_prover, &mut ark_std::test_rng());

        // take the same evaluation stream
        let len = 1 << NUM_VARIABLES;
        let evals: Vec<SmallGoldilocks> = (0..len).map(|x| SmallGoldilocks::from(x as u32)).collect();
        let mut fs = BenchFiatShamir::<Fp2SmallGoldilocks, _>::new(ark_std::test_rng());
        let transcript = prove(&evals, &mut fs);

        // were the challenges the same?
        let verifier_messages_fp4: Vec<Fp2SmallGoldilocks> = time_prover_transcript
            .verifier_messages
            .iter()
            .map(|a| Fp2SmallGoldilocks::from_base_prime_field(*a))
            .collect();
        assert_eq!(
            verifier_messages_fp4, transcript.verifier_messages,
            "challenges not same"
        );

        let prover_messages_fp4: Vec<(Fp2SmallGoldilocks, Fp2SmallGoldilocks)> = time_prover_transcript
            .prover_messages
            .iter()
            .map(|&(a, b)| {
                (
                    Fp2SmallGoldilocks::from_base_prime_field(a),
                    Fp2SmallGoldilocks::from_base_prime_field(b),
                )
            })
            .collect();
        assert_eq!(
            prover_messages_fp4, transcript.prover_messages,
            "prover messages not same"
        );
    }
}
