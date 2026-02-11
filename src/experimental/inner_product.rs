use ark_std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;

use ark_ff::Field;

use crate::{
    multilinear::{reductions::pairwise, ReduceMode},
    multilinear_product::{TimeProductProver, TimeProductProverConfig},
    prover::Prover,
    streams::MemoryStream,
    ProductSumcheck,
};

use crate::experimental::transcript::Transcript;

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

pub fn inner_product<F: Field>(
    f: &mut Vec<F>,
    g: &mut Vec<F>,
    transcript: &mut impl Transcript<F>,
) -> ProductSumcheck<F> {
    // checks
    assert_eq!(f.len(), g.len());
    assert!(f.len().count_ones() == 1);

    // initialize
    let num_rounds = f.len().trailing_zeros() as usize;
    let mut prover_messages: Vec<(F, F, F)> = vec![];
    let mut verifier_messages: Vec<F> = vec![];

    // all rounds
    for _ in 0..num_rounds {
        let mut prover = TimeProductProver::new(TimeProductProverConfig::new(
            f.len().trailing_zeros() as usize,
            vec![MemoryStream::new(f.to_vec()), MemoryStream::new(g.to_vec())],
            ReduceMode::Pairwise,
        ));

        // call the prover
        let msg = prover.next_message(None).unwrap();

        // write transcript
        prover_messages.push(msg);
        transcript.write(msg.0);
        transcript.write(msg.1);
        transcript.write(msg.2);

        // read the transcript
        let chg = transcript.read();
        verifier_messages.push(chg);

        // reduce
        pairwise::reduce_evaluations(f, chg);
        pairwise::reduce_evaluations(g, chg);
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

    // Use F64 from the existing test fields
    use crate::tests::F64;

    const NUM_VARS: usize = 4; // vectors of length 2^4 = 16

    #[test]
    fn test_inner_product_sanity() {
        use crate::experimental::transcript::SanityTranscript;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        let mut transcript = SanityTranscript::new(&mut rng);
        let result = inner_product(&mut f, &mut g, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }

    #[test]
    fn test_inner_product_spongefish() {
        use crate::experimental::transcript::SpongefishTranscript;
        use spongefish::codecs::arkworks_algebra::FieldDomainSeparator;
        use spongefish::DomainSeparator;

        let mut rng = test_rng();

        let n = 1 << NUM_VARS;
        let mut f: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();
        let mut g: Vec<F64> = (0..n).map(|_| F64::rand(&mut rng)).collect();

        // Build the IO pattern: each round absorbs 3 scalars and squeezes 1 challenge
        let mut domsep = DomainSeparator::new("test-inner-product");
        for _ in 0..NUM_VARS {
            domsep =
                <DomainSeparator as FieldDomainSeparator<F64>>::add_scalars(domsep, 3, "prover");
            domsep = <DomainSeparator as FieldDomainSeparator<F64>>::challenge_scalars(
                domsep, 1, "verifier",
            );
        }

        let prover_state = domsep.to_prover_state();
        let mut transcript = SpongefishTranscript::new(prover_state);
        let result = inner_product(&mut f, &mut g, &mut transcript);

        assert_eq!(result.prover_messages.len(), NUM_VARS);
        assert_eq!(result.verifier_messages.len(), NUM_VARS);
    }
}
