use ark_ff::Field;

pub fn multivariate_claim<F: Field, S: Stream<F>>(stream: S) -> F {
    let mut claim = F::zero();
    let num_vars = stream.num_variables();
    let num_evaluations = 2usize.pow(num_vars as u32);

    for i in 0..num_evaluations {
        let eval = stream.evaluation(i);
        claim += eval * eval;
    }

    claim
}

pub fn multivariate_product_claim<F: Field, S: Stream<F>>(streams: Vec<S>) -> F {
    // should be given at least one stream
    let number_of_streams = streams.len();
    assert!(number_of_streams > 0);

    // all streams should have the same number of variables
    let num_vars = streams[0].num_variables();
    for stream in streams.iter() {
        assert_eq!(stream.num_variables(), num_vars);
    }

    // calculate the claim
    let mut claim = F::zero();
    let num_evaluations = 2usize.pow(num_vars as u32);
    for i in 0..num_evaluations {
        let mut inner_sum = F::one();
        for stream in streams.iter() {
            inner_sum *= stream.evaluation(i);
        }
        claim += inner_sum;
    }

    claim
}

pub trait Stream<F: Field>: Clone + Send + Sync {
    fn evaluation(&self, point: usize) -> F;
    fn num_variables(&self) -> usize;
}
