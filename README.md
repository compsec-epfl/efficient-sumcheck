<h1 align="center">Efficient Sumcheck</h1>

Efficient, streaming capable, sumcheck with **Fiat–Shamir** support via [SpongeFish](https://github.com/arkworks-rs/spongefish).

**DISCLAIMER:** This library has not undergone a formal security audit. If you’d like to coordinate an audit, please contact.

## General Use

This library exposes two high-level functions:
1) [`multilinear_sumcheck`](multilinear_sumcheck) and
2) [`inner_product_sumcheck`](inner_product_sumcheck).

Using [SpongeFish](https://github.com/arkworks-rs/spongefish) (or similar Fiat-Shamir interface) simply call the functions with the prover state:

### Multilinear Sumcheck
$claim = \sum_{x \in \{0,1\}^n} p(x)$
```rust
use efficient_sumcheck::{multilinear_sumcheck, Sumcheck};
use efficient_sumcheck::transcript::SanityTranscript;

let mut evals_p_01n: Vec<F> = /* ... */;
let mut prover_state = SanityTranscript::new(&mut rng);
let sumcheck_transcript: Sumcheck<F> = multilinear_sumcheck(
  &mut evals_p_01n,
  &mut prover_state
);
```

### Inner Product Sumcheck
$claim = \sum_{x \in \{0,1\}^n} f(x) \cdot g(x)$

```rust
use efficient_sumcheck::{inner_product_sumcheck, ProductSumcheck};
use efficient_sumcheck::transcript::SanityTranscript;

let mut evals_f_01n: Vec<F> = /* ... */;
let mut evals_g_01n: Vec<F> = /* ... */;
let mut prover_state = SanityTranscript::new(&mut rng);
let sumcheck_transcript: ProductSumcheck<F> = inner_product_sumcheck(
  &mut evals_f_01n,
  &mut evals_g_01n,
  &mut prover_state
);
```

## Showcase: WARP Multilinear Constraint Batching

[WARP](https://github.com/compsec-epfl/warp) is an IVC scheme that batches multilinear evaluation claims into a single inner product sumcheck. Before this library, WARP maintained its own 100+ line `MultilinearConstraintBatchingSumcheck` — a hand-rolled sumcheck loop with manual spongefish calls, pairwise reductions, and sparse-map folding ([PR #14](https://github.com/compsec-epfl/warp/pull/14)). All of that reduces to:

```rust
use efficient_sumcheck::{inner_product_sumcheck, batched_constraint_poly};

let alpha = inner_product_sumcheck(
    &mut f,
    &mut batched_constraint_poly(&dense_evals, &sparse_evals),
    &mut transcript,
).verifier_messages;
```

`batched_constraint_poly` merges **dense** evaluation vectors (e.g. out-of-domain sample queries) with **sparse** index-keyed corrections (e.g. in-domain shift queries optimized via [[CBBZ23](#references)]) into a single constraint polynomial, ready for the inner product sumcheck.

## Advanced Usage

Supporting the high-level interfaces are raw implementations of sumcheck [[LFKN92](#references)] using three proving algorithms:

- The quasi-linear time and logarithmic space algorithm of [[CTY11](#references)] 
  - [SpaceProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear/provers/space/space.rs#L8)
  - [SpaceProductProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_product/provers/space/space.rs#L11)

- The linear time and linear space algorithm of [[VSBW13](#references)] 
  - [TimeProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear/provers/time/time.rs#L13)
  - [TimeProductProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_product/provers/time/time.rs#L10)

- The linear time and sublinear space algorithms of [[CFFZ24](#references)] and [[BCFFMMZ25](#references)] respectively
  - [BlendyProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear/provers/blendy/blendy.rs#L9)
  - [BlendyProductProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_product/provers/blendy/blendy.rs#L13)

## References
[[LFNK92](https://dl.acm.org/doi/pdf/10.1145/146585.146605)]: Carsten Lund, Lance Fortnow, Howard J. Karloff, and Noam Nisan. “Algebraic Methods for Interactive Proof Systems”. In: Journal of the ACM 39.4 (1992).

[[CTY11](https://arxiv.org/pdf/1109.6882.pdf)]: Graham Cormode, Justin Thaler, and Ke Yi. “Verifying computations with streaming interactive proofs”. In: Proceedings of the VLDB Endowment 5.1 (2011), pp. 25–36.

[[VSBW13](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6547112)]: Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. “A hybrid architecture for interactive verifiable computation”. In: Proceedings of the 34th IEEE Symposium on Security and Privacy. Oakland ’13. 2013, pp. 223–237.

[[CFFZ24](https://eprint.iacr.org/2024/524.pdf)]: Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Andrew Zitek-Estrada. "A time-space tradeoff for the sumcheck prover". In: Cryptology ePrint Archive.

[[BCFFMMZ25](https://eprint.iacr.org/2025/1473.pdf)]: Anubhav Bawejal, Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Pratyush Mishra, Tushar Mopuri, and Andrew Zitek-Estrada. "Time-Space Trade-Offs for Sumcheck". In: TCC Theory of Cryptography: 23rd International Conference, pp. 37.