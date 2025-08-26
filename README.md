<h1 align="center">Efficient Sumcheck</h1>

<p align="center">
    <a href="https://github.com/compsec-epfl/space-efficient-sumcheck/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-APACHE-blue.svg"></a>
    <a href="https://github.com/compsec-epfl/space-efficient-sumcheck/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

This library was developed using [arkworks](https://arkworks.rs) to accompany:

- [Time-Space Trade-Offs for Sumcheck](https://eprint.iacr.org/2025/1473)<br>
[Anubhav Baweja](https://dblp.org/pid/192/1642), [Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Elisabetta Fedele](https://elisabettafedele.github.io), [Giacomo Fenzi](https://gfenzi.io), [Pratyush Mishra](https://pratyushmishra.com/), [Tushar Mopuri](https://tmopuri.com/) and [Andrew Zitek-Estrada](https://github.com/z-tech)

- [A Time-Space Tradeoff for the Sumcheck Prover](eprint.iacr.org/2024/524)<br>
[Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Elisabetta Fedele](https://elisabettafedele.github.io), [Giacomo Fenzi](https://gfenzi.io), and [Andrew Zitek-Estrada](https://github.com/z-tech)

It is a repository of algorithms and abstractions including but not limited Blendy 🍹.

**DISCLAIMER:** This library has not received security review and is NOT recommended for production use.

## Overview
The library provides implementation of sumcheck [[LFKN92](#references)] including product sumcheck. For adaptability to different contexts, it implements three proving algorithms:

- The quasi-linear time and logarithmic space algorithm of [[CTY11](#references)] 
  - [SpaceProver](./src/SpaceProver)
  - [SpaceProductProver](./src/SpaceProductProver)

- The linear time and linear space algorithm of [[VSBW13](#references)] 
  - [TimeProver](./src/TimeProver)
  - [TimeProductProver](./src/TimeProductProver)

- The linear time and sublinear space algorithm Blendy🍹
  - [BlendyProver](./src/BlendyProver)
  - [BlendyProductProver](./src/BlendyProductProver)

##  Usage
The library can be used to obtain a sumcheck transcript over any implementation of [Stream](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/streams/stream.rs#L41), which could be backed by an evaluations table held in memory or read from disk. For example, if $f = 4x_1x_2 + 7x_2x_3 + 2x_1 + 13x_2$ like in the test [here](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/tests/polynomials.rs#L15), then:

```
let f_stream: MemoryStream<F> = MemoryStream::<F>::new(f.to_evaluations());
let mut multivariate_prover = TimeProver::<F, MemoryStream<F>>::new(
    <TimeProver<F, MemoryStream<F>> as Prover<F>>::ProverConfig::default(
        f_stream.claimed_sum,
        3,
        p_stream,
    ),
);
let transcript = Sumcheck::<F>::prove::<
    MemoryStream<F>,
    TimeProver<F, MemoryStream<F>>,
>(&mut multivariate_prover, &mut ark_std::test_rng()));
```

Or for the sum of $f * g$, then:
```
let f_stream: MemoryStream<F> = MemoryStream::<F>::new(f.to_evaluations());
let g_stream: MemoryStream<F> = MemoryStream::<F>::new(g.to_evaluations());
let streams: Vec<MemoryStream<F>> = vec![f_stream, g_stream]; 
let multivariate_product_prover = TimeProductProver::<F, MemoryStream<F>>::new(ProductProverConfig::default(
    multivariate_product_claim(streams.clone()),
    num_vars,
    streams,
));
```

## Evaluation
In addition to the reference papers, to help selection of prover algorithm we give a brief evaluation. The asymptotic improvement of BlendyProver translates to significantly lower memory consumption than TimeProver across all configurations tested. TimeProver and BlendyProver have similar runtimes and are orders of magnitude faster than SpaceProver.

<p align="center">
    <img src="assets/evaluation_graphs.png#gh-light-mode-only" alt="Line graph showing runtime and memory consumption of provers for inputs ranging from 15 to 30 variables" style="max-width: 800px;" />
    <img src="assets/evaluation_graphs_inverted.png#gh-dark-mode-only" alt="Line graph showing runtime and memory consumption of provers for inputs ranging from 15 to 30 variables" style="max-width: 800px;" />
</p>

## Contribution
Contributions in the form of PRs and issues/ suggestions are welcome.

## References
[[LFNK92](https://dl.acm.org/doi/pdf/10.1145/146585.146605)]: Carsten Lund, Lance Fortnow, Howard J. Karloff, and Noam Nisan. “Algebraic Methods for Interactive Proof Systems”. In: Journal of the ACM 39.4 (1992).

[[CTY11](https://arxiv.org/pdf/1109.6882.pdf)]: Graham Cormode, Justin Thaler, and Ke Yi. “Verifying computations with streaming interactive proofs”. In: Proceedings of the VLDB Endowment 5.1 (2011), pp. 25–36.

[[VSBW13](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6547112)]: Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. “A hybrid architecture for interactive verifiable computation”. In: Proceedings of the 34th IEEE Symposium on Security and Privacy. Oakland ’13. 2013, pp. 223–237.
