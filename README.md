<h1 align="center">Blendy 🍹: a space-efficient sumcheck algorithm</h1>

<p align="center">
    <a href="https://github.com/compsec-epfl/space-efficient-sumcheck/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-APACHE-blue.svg"></a>
    <a href="https://github.com/compsec-epfl/space-efficient-sumcheck/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

This library was developed using the [arkworks](https://arkworks.rs) ecosystem to accompany [A Time-Space Tradeoff for the Sumcheck Prover](https://eprint.iacr.org/2024/524). 
By [Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Elisabetta Fedele](https://elisabettafedele.github.io), [Giacomo Fenzi](https://gfenzi.io), and [Andrew Zitek-Estrada](https://github.com/z-tech).

**WARNING:** This is an academic prototype and has not received careful code review. This implementation is NOT ready for production use

## Overview
This library provides implementation of the sumcheck algorithm [[LFKN92](#references)]. We focus on the implementation of the prover, and implement three algorithms:
- The quasi-linear time and logarithmic space algorithm of [[CTY11](#references)]
- The linear time and linear space algorithm of [[VSBW13](#references)]
- Our new algorithm: Blendy🍹 which runs in linear time and sublinear space.

## Modules
[Prover](/src/provers/prover.rs) is a trait implemented by each algorithm<br>
[SpaceProver](/src/provers/space_prover.rs) implements [[CTY11](#references)]<br>
[TimeProver](/src/provers/time_prover.rs) implements [[VSBW13](#references)]<br>
[BlendyProver](/src/provers/blendy_prover.rs) implements Blendy🍹<br>
[Proof](/src/proof.rs) is a runner that takes a Prover and randomness and runs the protocol to generate a transcript<br>
[Lag Poly](/src/provers/lagrange_polynomial.rs) implements the sequential lagrange polynomial routine described in section 4.1<br>
[Hypercube](/src/provers/hypercube.rs) implements `iter` for boolean hypercube members as a wrapper over a `usize`<br>

## Evaluation
We perform an evaluation of the three algorithms we implemented. The asymptotic improvement of BlendyProver translates to significantly lower memory consumption than TimeProver across all configurations tested. TimeProver and BlendyProver have similar runtimes and are orders of magnitude faster than SpaceProver.

<p align="center">
    <img src="assets/evaluation_graphs.png#gh-light-mode-only" alt="Line graph showing runtime and memory consumption of provers for inputs ranging from 15 to 30 variables" style="max-width: 800px;" />
    <img src="assets/evaluation_graphs_inverted.png#gh-dark-mode-only" alt="Line graph showing runtime and memory consumption of provers for inputs ranging from 15 to 30 variables" style="max-width: 800px;" />
</p>

##  Usage
The library can be used to run sumcheck over any implementation for [Stream](/src/provers/evaluation_stream.rs) to obtain a transcript: 

    use ark_std::rand::Rng;
    use space_efficient_sumcheck::{
        provers::{
            test_helpers::BenchStream, BlendyProver, Prover, ProverArgs,
        },
        Sumcheck,
    };


    let mut rng = ark_std::test_rng();
    let stream: BenchStream<TestField> = BenchStream::new();
    let transcript = Sumcheck::prove(
        &mut BlendyProver::<F>::new(BlendyProver::generate_default_args(
            Box::new(&stream),
        )),
        &mut rng,
    );

## License
This library is released under the MIT and Apache v2 Licenses.

## Paper
[A Time-Space Tradeoff for the Sumcheck Prover](eprint.iacr.org/2024/524)<br>
[Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Elisabetta Fedele](https://elisabettafedele.github.io), [Giacomo Fenzi](https://gfenzi.io), and [Andrew Zitek-Estrada](https://github.com/z-tech)

## References
[[LFNK92](https://dl.acm.org/doi/pdf/10.1145/146585.146605)]: Carsten Lund, Lance Fortnow, Howard J. Karloff, and Noam Nisan. “Algebraic Methods for Interactive Proof Systems”. In: Journal of the ACM 39.4 (1992).

[[CTY11](https://arxiv.org/pdf/1109.6882.pdf)]: Graham Cormode, Justin Thaler, and Ke Yi. “Verifying computations with streaming interactive proofs”. In: Proceedings of the VLDB Endowment 5.1 (2011), pp. 25–36.

[[VSBW13](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6547112)]: Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. “A hybrid architecture for interactive verifiable computation”. In: Proceedings of the 34th IEEE Symposium on Security and Privacy. Oakland ’13. 2013, pp. 223–237.
