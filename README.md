<h1 align="center">Efficient Sumcheck</h1>

Efficient, streaming capable, sumcheck with **Fiat–Shamir** support via [SpongeFish](https://github.com/arkworks-rs/spongefish).

**Security note:** This library has not undergone a formal security audit.

## General Use

This library exposes two high-level functions:
1) [`multilinear_sumcheck`](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_sumcheck.rs#L123) and
2) [`inner_product_sumcheck`](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/inner_product_sumcheck.rs#L166).

Both are parameterized by two field types: `BF` (base field, of the evaluations) and `EF` (extension field, of the challenges). When no extension field is needed, set `EF = BF`.

Using [SpongeFish](https://github.com/arkworks-rs/spongefish) (or similar Fiat-Shamir interface) simply call the functions with the prover state:

### Multilinear Sumcheck
$claim = \sum_{x \in \{0,1\}^n} p(x)$
```rust
use efficient_sumcheck::{multilinear_sumcheck, Sumcheck};
use efficient_sumcheck::transcript::SanityTranscript;

let mut evals_p_01n: Vec<BF> = /* ... */;
let mut prover_state = SanityTranscript::new(&mut rng);
let sumcheck_transcript: Sumcheck<EF> = multilinear_sumcheck::<BF, EF>(
  &mut evals_p_01n,
  &mut prover_state
);
```

### Inner Product Sumcheck
$claim = \sum_{x \in \{0,1\}^n} f(x) \cdot g(x)$

```rust
use efficient_sumcheck::{inner_product_sumcheck, ProductSumcheck};
use efficient_sumcheck::transcript::SanityTranscript;

let mut evals_f_01n: Vec<BF> = /* ... */;
let mut evals_g_01n: Vec<BF> = /* ... */;
let mut prover_state = SanityTranscript::new(&mut rng);
let sumcheck_transcript: ProductSumcheck<EF> = inner_product_sumcheck::<BF, EF>(
  &mut evals_f_01n,
  &mut evals_g_01n,
  &mut prover_state
);
```

## Examples

### 1) WARP - Multilinear Constraint Batching

Before integration, [WARP](https://github.com/compsec-epfl/warp) used 200+ lines of sumcheck related code including calls to SpongeFish, pair- and table-wise reductions, as well as sparse-map foldings ([PR #14](https://github.com/compsec-epfl/warp/pull/14), [PR #12](https://github.com/compsec-epfl/warp/pull/12/changes#diff-904f410986c619441fb8554f4840cb36613f2de354b41ca991d381dec78959b0L34)). 

Using Efficient Sumcheck this reduces to six lines of code and brings parallelization via Rayon (and soon vectorization via SIMD):

```rust
use efficient_sumcheck::{inner_product_sumcheck, batched_constraint_poly};

let alpha = inner_product_sumcheck::<BF, EF>(
    &mut f,
    &mut batched_constraint_poly(&dense_evals, &sparse_evals),
    &mut transcript,
).verifier_messages;
```

Here, `batched_constraint_poly` merges dense evaluation vectors (out-of-domain samples) with sparse map-represented polynomials (in-domain queries) into a single constraint polynomial, ready for the inner product sumcheck.

## Advanced Usage

Supporting the high-level interfaces are raw implementations of sumcheck [[LFKN92](#references)] using three proving algorithms:

- The quasi-linear time and logarithmic space algorithm of [[CTY11](#references)] 
  - [SpaceProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear/provers/space/core.rs#L8)
  - [SpaceProductProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_product/provers/space/core.rs#L11)

- The linear time and linear space algorithm of [[VSBW13](#references)] 
  - [TimeProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear/provers/time/core.rs#L7)
  - [TimeProductProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_product/provers/time/core.rs#L16)

- The linear time and sublinear space algorithms of [[CFFZ24](#references)] and [[BCFFMMZ25](#references)] respectively
  - [BlendyProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear/provers/blendy/core.rs#L14)
  - [BlendyProductProver](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_product/provers/blendy/core.rs#L13)

## References
[[LFNK92](https://dl.acm.org/doi/pdf/10.1145/146585.146605)]: Carsten Lund, Lance Fortnow, Howard J. Karloff, and Noam Nisan. “Algebraic Methods for Interactive Proof Systems”. In: Journal of the ACM 39.4 (1992).

[[CTY11](https://arxiv.org/pdf/1109.6882.pdf)]: Graham Cormode, Justin Thaler, and Ke Yi. “Verifying computations with streaming interactive proofs”. In: Proceedings of the VLDB Endowment 5.1 (2011), pp. 25–36.

[[VSBW13](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6547112)]: Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. “A hybrid architecture for interactive verifiable computation”. In: Proceedings of the 34th IEEE Symposium on Security and Privacy. Oakland ’13. 2013, pp. 223–237.

[[CFFZ24](https://eprint.iacr.org/2024/524.pdf)]: Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Andrew Zitek-Estrada. "A time-space tradeoff for the sumcheck prover". In: Cryptology ePrint Archive.

[[BCFFMMZ25](https://eprint.iacr.org/2025/1473.pdf)]: Anubhav Bawejal, Alessandro Chiesa, Elisabetta Fedele, Giacomo Fenzi, Pratyush Mishra, Tushar Mopuri, and Andrew Zitek-Estrada. "Time-Space Trade-Offs for Sumcheck". In: TCC Theory of Cryptography: 23rd International Conference, pp. 37.
