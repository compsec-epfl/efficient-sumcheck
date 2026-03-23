<h1 align="center">Efficient Sumcheck</h1>

Efficient, streaming capable, sumcheck with **Fiat–Shamir** support via [SpongeFish](https://github.com/arkworks-rs/spongefish).

**Security note:** This library has not undergone a formal security audit.

## General Use

This library exposes three high-level functions:
1) [`multilinear_sumcheck`](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/multilinear_sumcheck.rs#L123),
2) [`inner_product_sumcheck`](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/inner_product_sumcheck.rs#L166), and
3) [`coefficient_sumcheck`](https://github.com/compsec-epfl/efficient-sumcheck/blob/main/src/coefficient_sumcheck.rs#L17).

The first two are parameterized by two field types: `BF` (base field, of the evaluations) and `EF` (extension field, of the challenges). When no extension field is needed, set `EF = BF`.

Using [SpongeFish](https://github.com/arkworks-rs/spongefish) (or similar Fiat-Shamir interface) simply call the functions with the Spongefish transcript:

### Multilinear Sumcheck
```math
claim = \sum_{x \in \{0,1\}^n} p(x)
```
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
```math
claim = \sum_{x \in \{0,1\}^n} f(x) \cdot g(x)
```

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

### Coefficient Sumcheck
```math
claim = \sum_{x \in \{0,1\}^n} p(x), \quad \deg_{x_i}(p) \leq d
```

Unlike the multilinear and inner product variants where `p` is multilinear (degree 1 in each variable, yielding degree-1 round polynomials), `coefficient_sumcheck` handles polynomials with arbitrary per-variable degree `d`, producing degree-`d` round polynomials. The user supplies a closure `compute_round_poly` that computes each round polynomial; the library handles transcript interaction and table reductions (both pairwise and tablewise) automatically.

```rust
use efficient_sumcheck::coefficient_sumcheck::{coefficient_sumcheck, CoefficientSumcheck};
use efficient_sumcheck::transcript::SanityTranscript;
use ark_poly::univariate::DensePolynomial;

let mut tablewise: Vec<Vec<Vec<F>>> = /* multi-column tables */;
let mut pairwise: Vec<Vec<F>> = /* flat evaluation vectors */;
let mut transcript = SanityTranscript::new(&mut rng);

let result: CoefficientSumcheck<F> = coefficient_sumcheck(
  |tablewise, pairwise| {
      // Compute h(X) as a DensePolynomial<F> from current table state.
      // Return coefficients in ascending order: [c0, c1, ..., cd].
      DensePolynomial::from_coefficients_vec(vec![/* ... */])
  },
  &mut tablewise,
  &mut pairwise,
  n_rounds,
  &mut transcript,
);
```

The closure receives immutable references to the current tables; after each round the library automatically reduces all pairwise and tablewise entries by folding with the verifier challenge.

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

### 2) WARP - Twin Constraint Batching

[WARP](https://github.com/compsec-epfl/warp) also uses `coefficient_sumcheck` with `folding::protogalaxy::fold` to batch a codeword check and an R1CS constraint check into a single sumcheck. The codewords, witness vectors, and folding coefficients are stored as tablewise tables and the equality polynomial evaluations as a pairwise vector:

```rust
use efficient_sumcheck::coefficient_sumcheck::coefficient_sumcheck;
use efficient_sumcheck::folding::protogalaxy;

let mut tablewise = [codewords, z_vecs, alpha_vecs, beta_vecs];
let mut pairwise = [tau_eq_evals];

let sc = coefficient_sumcheck(
    |tw, pw| {
        let (u, z, a, b) = (&tw[0], &tw[1], &tw[2], &tw[3]);
        let tau = &pw[0];

        let f = protogalaxy::fold(/* ... */, /* codeword polys */);
        let p = protogalaxy::fold(/* ... */, /* constraint polys */);
        let t = linear_poly(tau[0], tau[1]);

        // h(X) = (f(X) + ω·p(X)) · t(X)
        (f + p * omega).naive_mul(&t)
    },
    &mut tablewise,
    &mut pairwise,
    log_l,
    &mut prover_state,
);
let gamma = sc.verifier_messages;
```

After each round `coefficient_sumcheck` reduces all four tablewise tables and the pairwise equality evaluations by folding with the verifier's challenge.

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
