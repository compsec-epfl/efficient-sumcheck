# Sumcheck API Design

Authoritative reference: Justin Thaler, *Proofs, Arguments, and Zero-Knowledge*,
Chapter 4 ("Interactive Proofs"), July 2023.
https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf

## 1. The protocol (Thaler &sect;4.1)

Given a v-variate polynomial g over a finite field F with degree at most d
in each variable, the sum-check protocol proves:

```
H = sum_{b in {0,1}^v} g(b_1, ..., b_v)
```

The protocol proceeds in v rounds. Each round j:

1. **P -> V**: a univariate polynomial g_j(X_j) of degree <= deg_j(g),
   specified by its evaluations at {0, 1, ..., deg_j(g)}.
2. **V checks**: g_{j-1}(r_{j-1}) = g_j(0) + g_j(1).
   (For round 1, checks H = g_1(0) + g_1(1).)
3. **V -> P**: random r_j in F.

After round v: V checks g_v(r_v) = g(r_1, ..., r_v) via an oracle query
(or by delegation to another protocol).

**Proposition 4.1.** Completeness error 0, soundness error <= vd / |F|.

### What the prover sends

Table 4.1: total prover-to-verifier communication is
sum_{j=1}^{v} (deg_j(g) + 1) field elements. When deg_j(g) = O(1) for
all j (the common case), this is O(v) field elements.

### What the verifier checks

Per round: one consistency equation (g_j(0) + g_j(1) = previous claim)
and a degree bound (deg g_j <= deg_j(g)). After all rounds: one oracle
query to g at a random point.

Total verifier time: O(v + sum_j deg_j(g)) + T, where T is the cost of
one evaluation of g at a point in F^v.

## 2. Our three use cases are one protocol

Thaler's protocol is parameterized by g. Our current three entry points
are three instantiations:

| Use case                | g                           | deg per var | Ref        |
|-------------------------|-----------------------------|-------------|------------|
| multilinear_sumcheck    | f_tilde (MLE of evals)      | 1           | &sect;4.1  |
| inner_product_sumcheck  | f_tilde * g_tilde           | 2           | &sect;4.4  |
| coefficient_sumcheck    | user-defined                | d           | &sect;4.6  |

In all three cases, the *protocol* is identical &mdash; only the prover's
round-polynomial computation changes. This motivates a single protocol
runner parameterized by a prover trait, not three separate functions.

## 3. The prover trait

```rust
/// Prover side of the sum-check protocol (Thaler &sect;4.1).
///
/// Implementors define how the round polynomial g_j is computed
/// from the prover's internal state. The protocol runner calls
/// `round()` once per round, then the caller inspects post-state.
pub trait SumcheckProver<F: SumcheckField> {
    /// Degree of g_j in the current variable X_j.
    fn degree(&self) -> usize;

    /// Compute g_j and advance state.
    ///
    /// Returns evaluations of g_j at {0, 1, ..., degree()}.
    /// `challenge` is `None` for round 0 (no prior challenge exists);
    /// `Some(r_{j-1})` for rounds j >= 1, used to fold internal state.
    fn round(&mut self, challenge: Option<F>) -> Vec<F>;

    /// Apply the final verifier challenge.
    /// Called once after the last round, before `final_value()`.
    fn finalize(&mut self, last_challenge: F);

    /// After finalize(): the claimed value g(r_1, ..., r_v).
    fn final_value(&self) -> F;
}
```

The prover is passed as `&mut P` to the protocol runner. After sumcheck
completes, the caller retains ownership and can query prover-specific
post-state (e.g., for GKR: the two claimed W values at b* and c*).

## 4. Prover strategies and streaming

### Two axes: space strategy and variable ordering

The prover has two independent design choices:

1. **Space strategy** &mdash; how much memory to budget:
   - *Time*: O(2^v) space, O(2^v) total time. Holds all evaluations.
   - *Blendy*: O(2^k) space, O(2^v) total time (k &lt; v). Partitions
     variables into stages of size k, recomputes per stage.
   - *Space*: O(v) space, O(v &middot; 2^v) total time. Academic only.

2. **Variable ordering** &mdash; which variable to fold each round:
   - *MSB* (half-split): fold the topmost variable. Pairs `(v[k], v[k+L/2])`.
     Best for in-memory and random-access-streaming workloads.
   - *LSB* (pair-split): fold the bottommost variable. Pairs `(v[2k], v[2k+1])`.
     Best for sequential-streaming workloads where data arrives incrementally.

These choices are orthogonal: you can use blendy with MSB ordering (large
witness on SSD) or blendy with LSB ordering (Jolt CPU trace).

### Streaming taxonomy

The choice of variable ordering depends on how data is available:

| Scenario | Data availability | Access | Best ordering | Strategy |
|----------|-------------------|--------|---------------|----------|
| In-memory | Full table in RAM | Random | MSB | Time |
| Random-access stream | Exists on disk, too big for RAM | Seekable | MSB | Blendy |
| Sequential stream | Generated incrementally | Forward-only | LSB | Blendy |

**Random-access streaming** (e.g., large witness mmap'd from SSD): the data
exists but doesn't fit in RAM. MSB ordering has better cache behavior because
it reads two contiguous half-table regions; the blendy working set fits in
cache while the full table is paged in as needed.

**Sequential streaming** (e.g., Jolt CPU trace): evaluations are computed
on-the-fly and arrive in index order (0, 1, 2, ...). LSB ordering is optimal
because adjacent pairs `(f[2k], f[2k+1])` are immediately available &mdash;
the prover can begin folding before the full table exists.

In both streaming cases, blendy is used because the full table doesn't fit
in the working set. Blendy is the *space strategy*; MSB vs LSB is the
*traversal order*. They are independent.

### Blendy stage scheduling

Standard blendy (CFFZ24) partitions v variables into stages of fixed size k,
recomputing the partial-sum table once per stage. The optimal k depends on
the ratio of cache sizes to element size.

Jolt's `HalfSplitSchedule` (based on BCFFMMZ25, eprint 2025/1473) takes a
different approach: **cost-model-driven, non-uniform window sizes**.

For a degree-d sumcheck, the cost of processing a window of w variables
starting at round i is `(d+1)^w / 2^(w+i) * T` where T is the trace
length. Setting cost ~ 1 gives optimal window size:

```
w(i) = round(ratio * i)    where ratio = ln(2) / ln((d+1)/2)
```

This produces growing windows: early rounds use small windows (the
hypercube is large), later rounds use larger windows (the residual sum is
small). For degree 2, the windows grow as 1, 2, 5, 14, ...

| Degree | Ratio | Example window sequence |
|--------|-------|------------------------|
| 2 | 1.71 | 1, 2, 5, 14, ... |
| 3 | 1.00 | 1, 1, 2, 3, 4, ... |
| 4 | 0.76 | 1, 1, 1, 2, 2, 3, ... |

The schedule is parameterized by a `StreamingSchedule` trait, not a fixed
constant:

```rust
pub trait StreamingSchedule {
    fn num_rounds(&self) -> usize;
    fn switch_over_point(&self) -> usize;
    fn is_window_start(&self, round: usize) -> bool;
    fn num_unbound_vars(&self, round: usize) -> usize;
}
```

This allows tuning per deployment target and polynomial degree.

### Strategies table

Per Thaler &sect;4.4.3 and prior work (CTY11, VSBW13, CFFZ24, BCFFMMZ25):

| Strategy | Space     | Total time   | Input        | Ordering | Ref              |
|----------|-----------|--------------|--------------|----------|------------------|
| Time     | O(2^v)   | O(2^v)       | Vec\<F\>     | MSB      | VSBW13           |
| Blendy   | O(2^k)   | O(2^v)       | Stream\<F\>  | MSB or LSB | CFFZ24, BCFFMMZ25 |
| Space    | O(v)     | O(v * 2^v)   | Stream\<F\>  | MSB or LSB | CTY11            |

All implement `SumcheckProver<F>`. The difference is internal:
how `round()` computes the polynomial from the data.

### Construction

```rust
// In-memory (MSB, time strategy).
impl<F: SumcheckField> MultilinearProver<F> {
    pub fn new(evals: Vec<F>) -> Self;
}

// Streaming (LSB or MSB, blendy strategy).
impl<F: SumcheckField> StreamingMultilinearProver<F> {
    /// Random-access stream, MSB ordering. Best for mmap'd data.
    pub fn new_msb<S: Stream<F>>(stream: S, k: usize) -> Self;

    /// Sequential stream, LSB ordering. Best for incremental data (Jolt).
    pub fn new_lsb<S: Stream<F>>(stream: S, k: usize) -> Self;
}
```

The `Stream` trait (already in `src/streams/`) provides random access to
evaluations without requiring the full table in memory.

## 5. Three polynomial shapes

Each polynomial shape (multilinear, product, general) has its own prover
type for each strategy, but all implement `SumcheckProver<F>`:

```rust
/// g = f_tilde, degree 1. Prover folds evals via Lemma 4.3.
pub struct MultilinearProver<F> { evals: Vec<F> }

/// g = f_tilde * g_tilde, degree 2. Prover folds both vectors.
pub struct InnerProductProver<F> { a: Vec<F>, b: Vec<F> }

/// g = user-defined, degree d. Wraps a RoundPolyEvaluator.
pub struct CoefficientProver<F, E: RoundPolyEvaluator<F>> { ... }
```

### The time prover's round() (Lemma 4.3 and 4.5)

For a multilinear f over {0,1}^v with evaluations in array A:

```
A[x'] = r_1 * A[1, x'] + (1 - r_1) * A[0, x']
```

This is the **fold** operation (Lemma 4.3, equation 4.13). After folding,
the array has half the entries, and the prover can compute the next round
polynomial from the folded array.

For a product of k multilinears (Lemma 4.5), the same fold is applied to
each factor independently, and the round polynomial is computed from the
folded factors.

## 6. The fold primitive

The fold operation (Thaler Lemma 4.3, equation 4.13) is the computational
core of the time prover:

```rust
/// Half-split (MSB) fold: new[k] = v[k] + weight * (v[k + L/2] - v[k]).
///
/// Implicit zero padding for non-power-of-two inputs.
/// SIMD-accelerated for Goldilocks on NEON and AVX-512 IFMA.
pub fn fold<F: Field>(values: &mut Vec<F>, weight: F);
```

The fold is exposed as a standalone public function because callers
(e.g., WHIR's `multilinear_fold`) need it independently of the full
sumcheck protocol.

### Layout: half-split (MSB)

We fold the *top-most* variable each round: `v[0..L/2]` vs `v[L/2..L]`.
This is the MSB (most-significant-bit-first) convention. It matches
Thaler's equation 4.13 directly:

```
p(x_1, ..., x_l) = x_1 * p(1, x_2, ..., x_l) + (1 - x_1) * p(0, x_2, ..., x_l)
```

where `p(1, ...)` is the upper half and `p(0, ...)` is the lower half.

### SIMD opt-in for non-arkworks types (`SimdRepr`)

Non-arkworks Goldilocks implementations opt into SIMD via the `SimdRepr`
trait, whose memory layout guarantee is enforced at compile time by
`zerocopy`:

```rust
pub trait SimdRepr:
    SumcheckField + zerocopy::IntoBytes + zerocopy::FromBytes + zerocopy::Immutable
{
    fn modulus() -> u64;
}
```

The zerocopy bounds verify at derive time that the type supports safe byte
reinterpretation. No `unsafe` needed from the implementor. A wrong
`modulus()` produces wrong arithmetic (logic bug), not UB.

```rust
#[derive(zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
#[repr(transparent)]
struct JoltGoldilocks(u64);

impl SimdRepr for JoltGoldilocks {
    fn modulus() -> u64 { GOLDILOCKS_P }
}
```

Arkworks types bypass `SimdRepr` &mdash; the blanket `SumcheckField` impl
auto-detects Goldilocks from `BasePrimeField::MODULUS` and the detection
is const-folded by LLVM.

## 7. The protocol runner

```rust
/// Run the sum-check protocol for `num_rounds` rounds.
///
/// Partial execution (`num_rounds < v`) supports composed protocols
/// like GKR (one sumcheck per circuit layer) and WHIR (partial rounds
/// interleaved with commit/open).
///
/// `hook` is called each round after the prover message is written
/// and before the verifier challenge is read. Pass `|_, _| {}` when
/// no hook is needed.
pub fn sumcheck<F, T, H, P>(
    prover: &mut P,
    num_rounds: usize,
    transcript: &mut T,
    hook: H,
) -> SumcheckProof<F>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T),
    P: SumcheckProver<F>;
```

### The proof transcript

```rust
pub struct SumcheckProof<F: Field> {
    /// Round polynomial values per round, EvalsInfty wire format.
    /// `round_polys[j]` has `d = degree()` entries (see §7a).
    pub round_polys: Vec<Vec<F>>,
    /// Verifier challenges r_1, ..., r_v.
    pub challenges: Vec<F>,
    /// g(r_1, ..., r_v) -- the prover's claimed final evaluation.
    pub final_value: F,
}
```

The verifier can reconstruct any round's consistency check from
`round_polys` and `challenges`.

### 7a. Wire format: EvalsInfty

All provers emit round polynomials in **EvalsInfty** form — `d` values
per round for a degree-`d` polynomial, one fewer than a full
evaluation table:

| d | Wire | Verifier recovers |
|---|------|-------------------|
| 1 | `[h(0)]` | `h(1) = claim − h(0)` |
| ≥ 2 | `[h(0), h(∞), h(2), ..., h(d−1)]` | `h(1) = claim − h(0)` |

`h(∞)` is the leading coefficient (coefficient of `x^d`). The
verifier reconstructs `h(r)` by Lagrange-interpolating the
degree-`(d−1)` residual `q(x) = h(x) − h(∞)·x^d` over the finite
points `{0, 1, ..., d−1}` and then adding `h(∞)·r^d`.

**Why this shape.**

1. *Cheapest for product-structured summands.* The leading
   coefficient is a difference-of-shifted-evaluations form
   (see [BDDT25](https://eprint.iacr.org/2025/1117.pdf), Algorithm 3),
   which is the minimal additional work beyond computing `h(0)`.
2. *Consistency is structural.* The check `h(0) + h(1) = claim` is
   enforced by the wire format rather than a runtime equality test.
   The consequence is that a dishonest prover's misbehaviour no
   longer surfaces as a mid-protocol `ConsistencyCheck` error — it
   surfaces at the caller's oracle check, where `final_claim`
   diverges from `proof.final_value`. Soundness is preserved; the
   detection point moves by one step.
3. *One byte saved per round* (compared to sending `d + 1`
   evaluations), worth noting for transcript-size-sensitive callers.

The runner and the verifier are both written once against this
format; every concrete prover (multilinear, inner-product, GKR,
eq-factored, coefficient of any degree, MSB or LSB) produces it.

## 8. The verifier

One function that checks any sumcheck proof, regardless of the
polynomial's degree:

```rust
/// Verify a sum-check proof against a claimed sum.
///
/// Per round (EvalsInfty wire — see §7a): receives `d` values,
/// derives `h_j(1) = claim − h_j(0)` from the consistency constraint,
/// reconstructs the round polynomial from the received finite-point
/// evaluations plus the leading coefficient, and evaluates at the
/// challenge `r_j` to obtain the next round's claim.
///
/// Returns the final claimed value and the challenge vector on success.
/// The caller is responsible for the oracle check: verifying that
/// final_value = g(r_1, ..., r_v). How this is done depends on the
/// application (direct evaluation, delegation to another sumcheck,
/// polynomial commitment query, etc.).
pub fn sumcheck_verify<F, T, H>(
    claimed_sum: F,
    expected_degree: usize,
    num_rounds: usize,
    transcript: &mut T,
    hook: H,
) -> Result<(F, Vec<F>), SumcheckError>
where
    F: Field,
    T: Transcript<F>,
    H: FnMut(usize, &mut T);
```

The verifier does NOT perform the final oracle check. Per Remark 4.2,
the verifier can apply sumcheck "even without knowing the polynomial g."
The final check is the caller's responsibility because it depends on the
application:

- Standalone sumcheck: evaluate g(r) directly.
- GKR: delegate to the next layer's sumcheck.
- WHIR: check via polynomial commitment.
- MatMult: evaluate f_A and f_B at derived points.

## 9. Cross-field support (BF != EF)

In practice, evaluations often live in a small base field BF (e.g.,
Goldilocks, p = 2^64 - 2^32 + 1) while challenges are sampled from a
larger extension field EF (e.g., Goldilocks^3) for soundness.

This is a prover concern, not a protocol concern:

- Round 0: the prover computes g_1 over BF evaluations.
- After receiving r_1 in EF: the prover performs a cross-field fold,
  lifting BF data to EF.
- Rounds 1+: everything is in EF.

The `SumcheckProver` trait is generic over a single field `F` (= EF).
The BF -> EF transition happens inside the prover's `round()` method
when `challenge` transitions from `None` to `Some(r_1)`. The protocol
runner and verifier never see BF.

A convenience constructor handles the common case:

```rust
impl<BF: Field, EF: Field + From<BF>> MultilinearProver<EF> {
    /// Cross-field prover: evaluations in BF, challenges in EF.
    /// Round 0 computes in BF, then lifts to EF on first challenge.
    pub fn cross_field(evals: Vec<BF>) -> Self;
}
```

## 10. GKR compatibility (&sect;4.6)

GKR runs d sumcheck invocations (one per circuit layer). Each layer's
sumcheck is over a different polynomial f_r^(i) (equation 4.18):

```
f_r^(i)(b, c) = add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c))
              + mult_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))
```

This is a (2k_{i+1})-variate polynomial of degree 2 in each variable.

### What GKR needs from sumcheck

1. **Custom polynomial**: GKR defines its own round polynomial via the
   wiring predicates (add_i, mult_i). This is a custom `SumcheckProver`
   implementation -- the trait handles it.

2. **Partial execution**: each layer runs a full sumcheck (all rounds),
   but the *claim chains* between layers. The `sumcheck()` function
   returns, then GKR processes the result and starts a new sumcheck.

3. **Post-state inspection**: after sumcheck, GKR needs the prover's
   claimed values W_{i+1}(b*) and W_{i+1}(c*). Since `prover` is
   `&mut P`, the caller retains the prover and calls GKR-specific
   methods on the concrete type:

   ```rust
   let proof = sumcheck(&mut gkr_prover, num_rounds, &mut t, |_, _| {});
   let (w_b, w_c) = gkr_prover.claimed_w_values(); // GKR-specific method
   ```

4. **Reduce-to-one sub-protocol** (&sect;4.5.2, Claim 4.6): after each
   sumcheck, the verifier needs to reduce two evaluation claims to one.
   This is a separate one-round protocol, NOT part of sumcheck:

   ```rust
   /// Reduce claims W(b) = v_0 and W(c) = v_1 to a single claim
   /// W(r) = v at a random point on the line through b and c.
   pub fn reduce_to_one<F, T>(
       b: &[F], c: &[F],
       v0: F, v1: F,
       transcript: &mut T,
   ) -> (Vec<F>, F)
   ```

   This is a composable building block, not baked into sumcheck.

## 11. WHIR compatibility

WHIR's integration pattern:

```rust
for round_config in &self.round_configs {
    // Commit to the current folded polynomial
    round_config.committer.commit(&a);

    // OOD / in-domain queries, RLC into covector b
    update_covector(&mut b, &stir_challenges);

    // Partial sumcheck: fold a and b by folding_factor variables
    let proof = sumcheck(
        &mut InnerProductProver::new(a, b),
        round_config.folding_factor,
        &mut transcript,
        |_, t| round_config.round_pow.prove(t),
    );

    // Extract folded state for the next round
    a = prover.a();  // prover-specific accessor
    b = prover.b();
}
```

Key requirements satisfied:
- **Partial rounds**: `num_rounds = folding_factor < v`.
- **Hook**: proof-of-work grinding between write and read.
- **Post-state**: prover retains folded vectors after partial execution.
- **MSB fold**: `fold()` used independently for WHIR's `multilinear_fold`.

## 12. Jolt integration

Jolt (a16z/jolt) uses its own `SumcheckInstanceProver` trait, which
splits the round into two calls:

```rust
// Jolt's trait (simplified)
trait SumcheckInstanceProver<F, T> {
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F>;
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize);
    fn finalize(&mut self);
}
```

Our `SumcheckProver::round(challenge)` merges fold and compute into one
call. An adapter in Jolt's codebase bridges the two:

```rust
struct Adapter<P: SumcheckProver<F>> {
    inner: P,
    pending: Option<F>,
}

impl SumcheckInstanceProver<F, T> for Adapter<P> {
    fn compute_message(&mut self, _round: usize, _claim: F) -> UniPoly<F> {
        let c = self.pending.take();
        UniPoly::from_evals(&self.inner.round(c))
    }
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.pending = Some(r_j.into());
    }
    fn finalize(&mut self) {
        if let Some(c) = self.pending.take() {
            self.inner.finalize(c);
        }
    }
}
```

### Key compatibility points

- **Variable ordering**: Jolt defaults to LSB (`BindingOrder::LowToHigh`).
  Use `MultilinearProverLSB` for this path. Spartan outer sumcheck uses
  MSB (`HighToLow`) &mdash; use `MultilinearProver`.

- **Challenge type**: Jolt uses a narrow 128-bit `F::Challenge` type for
  performance. The adapter converts via `Into<F>` at the boundary.

- **Return type**: Jolt expects `UniPoly<F>` (coefficients). Our trait
  returns the EvalsInfty wire (§7a). Convert via reconstruction or
  have the prover return coefficients directly.

- **Batching**: Jolt's `BatchedSumcheck::prove` combines multiple instances
  with random linear combinations. The orchestrator handles batching &mdash;
  each instance just implements `SumcheckInstanceProver`.

- **Streaming schedule**: Jolt's `HalfSplitSchedule` (BCFFMMZ25) uses
  cost-model-driven window sizes. Our blendy implementation can adopt the
  same `StreamingSchedule` trait for stage sizing.

The adapter lives in Jolt's repository (or a thin integration crate), not
in this library. Our responsibility is keeping the trait surface clean
enough that the adapter is trivial.

## 13. Advanced optimizations (Bagad-Dao-Domb-Thaler, ePrint 2025/1117)

The `SumcheckProver` trait cleanly separates *what the protocol sends*
(round polynomials, final evaluation) from *how the prover computes them*.
All known prover-side optimizations live below this boundary and are
compatible without any trait or protocol changes.

### Algorithms 1--2: LinearTime and SqrtSpace

These are our "time" and "space" strategies (§4 above). LinearTime
maintains d arrays of size 2^(v−j), folding each round in O(2^v) total.
SqrtSpace streams over the input for the first v/2 rounds, using O(√2^v)
space. Both produce identical round polynomials. Each is a distinct
`SumcheckProver` implementation with different constructors
(`new` vs `from_stream`).

### Algorithms 3--4: SmallValue optimization

When sumcheck operates on a product g = f·eq(r, ·), many multiplications
involve "small" operands (those representable with few bits, e.g., R1CS
entries in {0, 1, −1}). The SmallValue optimization categorizes
multiplications as:

- **ss** (small × small): result fits in a machine word, no field reduction.
- **sl** (small × large): a few shifts and adds, no full mul.
- **ll** (large × large): full field multiplication.

Algorithms 3--4 defer ll multiplications by precomputing accumulators
and use Toom-Cook to reduce the number of ss multiplications from
3(d+1) to 2.5(d+1) per hypercube point.

**Trait compatibility**: this is entirely internal to `round()`. The prover
maintains richer internal state (categorized accumulator tables) and uses
cheaper multiplication routines, but returns the same EvalsInfty wire (§7a).
A `SmallValueProver<F>` implements `SumcheckProver<F>` with unchanged
`degree()` and `final_value()`.

### Algorithms 5--6: EqPoly optimization

For sumcheck over g(x) = eq(r, x) · h(x), the naive approach materializes
a 2^v-sized table for eq(r, ·). The EqPoly optimization splits
eq(r, x) = eq(r_L, x_L) · eq(r_R, x_R) and only maintains tables of size
2^(v/2), reducing ll multiplications from 2^v to 2^(v/2+1).

Algorithm 6 combines EqPoly with SmallValue for the Spartan-in-Jolt
use case.

**Trait compatibility**: this changes how the prover manages its internal
tables — specifically, it defers half the eq computation and processes it
in a blocked fashion. The round polynomial's degree and protocol messages
are unchanged. This would be a constructor variant or flag on the
prover type.

### Univariate skip (Gruen's optimization)

When g has degree > 1 in its first variable, the prover can compute
a univariate restriction t_j(X_j) = g(r_1, ..., r_{j-1}, X_j, x_{j+1}*, ...)
and derive the standard round polynomial s_j from it. Per Setty and
Thaler (Section 3.1 of the paper): "the sum-check verifier is unchanged."

This is relevant primarily in small-characteristic fields where the degree
of individual variables can be high. In our setting (Goldilocks, large
characteristic), the multilinear case (degree 1) doesn't benefit, but the
coefficient sumcheck (arbitrary degree d) could.

**Trait compatibility**: the prover computes the round polynomial via the
univariate-skip shortcut inside `round()`, but returns the same evaluations.
No change to `degree()`, no change to the verifier, no change to the
wire format.

### Per-round degree variation

Some optimizations work best when the polynomial's degree varies by round
(e.g., degree d in round 1, degree 1 in rounds 2--v for univariate skip).
Our current trait returns a single `degree()` value. If a future prover
needs per-round variation, the trait can evolve to:

```rust
fn degree(&self) -> usize;           // max degree (for allocation)
fn round_degree(&self) -> usize;     // degree of the current round
```

None of the paper's algorithms strictly require this — they maintain
constant degree throughout — but this is a natural extension point if
needed.

### Summary: all optimizations are prover-internal

| Optimization              | Changes protocol? | Changes wire format? | Fits `SumcheckProver`? |
|---------------------------|-------------------|----------------------|------------------------|
| LinearTime (Alg 1)        | No                | No                   | Yes — "time" strategy  |
| SqrtSpace (Alg 2)         | No                | No                   | Yes — "space" strategy |
| SmallValue (Alg 3--4)     | No                | No                   | Yes — smarter `round()` |
| EqPoly (Alg 5--6)         | No                | No                   | Yes — internal table mgmt |
| Univariate skip            | No                | No                   | Yes — alternate `round()` |

The trait is the abstraction boundary: everything above it (protocol runner,
transcript, verifier) stays fixed; everything below it (prover strategy,
SIMD kernels, table layout, multiplication tricks) is implementation
freedom.

## 14. What is NOT in scope

- **Generic IP trait**: sumcheck is a specific protocol, not an instance
  of a generic framework. The `Transcript` trait already captures the
  interaction pattern. If GKR or FRI reveal common structure, a
  shared trait can emerge later.

- **Zero-knowledge**: ZK sumcheck adds masking polynomials but uses the
  same protocol structure. A future `ZkSumcheckProver` wrapper could
  add masking without changing the protocol runner.

- **Batching**: multiple sumcheck instances can be batched with random
  linear combinations. This is a composition technique, not a protocol
  change.

- **Reduce-to-one**: a separate composable sub-protocol (&sect;4.5.2),
  not part of sumcheck itself.

## 15. Migration from current API

| Current                                   | New                                            |
|-------------------------------------------|------------------------------------------------|
| `multilinear_sumcheck(evals, t, hook)`    | `sumcheck(&mut MultilinearProver::new(evals), n, t, hook)` |
| `inner_product_sumcheck(a, b, t, hook)`   | `sumcheck(&mut InnerProductProver::new(a, b), n, t, hook)` |
| `multilinear_sumcheck_partial(..., k, h)` | `sumcheck(&mut prover, k, t, hook)`            |
| `fold(values, weight)`                    | `fold(values, weight)` (unchanged)             |
| `coefficient_sumcheck(...)`               | `sumcheck(&mut CoefficientProver::new(...), n, t, hook)` |
| `Sumcheck<F>`                             | `SumcheckProof<F>`                             |
| `ProductSumcheck<F>`                      | `SumcheckProof<F>` (unified)                   |
| `multilinear_sumcheck_verify(...)`        | `sumcheck_verify(sum, deg, n, t, hook)`        |

The `Sumcheck` and `ProductSumcheck` return types unify into one
`SumcheckProof<F>`. The `final_evaluations: (F, F)` field from
`ProductSumcheck` becomes a prover-specific accessor on
`InnerProductProver` post-state.

## 16. Benchmarking

### Three benchmark layers

**Layer 1: Kernel throughput (elements/second).** Measures the computational
core — fold and round polynomial evaluation — independent of the protocol.

- `fold` throughput: elements/sec for base field and each extension degree,
  with and without SIMD. Compare against theoretical memory bandwidth
  (Apple M-series: ~100 GB/s, Sapphire Rapids: ~50 GB/s per channel).
  A Goldilocks element is 8 bytes, so the ceiling for base-field fold on
  M2 is ~12.5 billion elements/sec.
- `round()` throughput: fold + evaluate combined. Shows the overhead of
  evaluation on top of fold.
- Cross-field promotion cost: measure the first EF round (BF→EF lift)
  separately. For ext3, memory triples and arithmetic cost jumps ~9×.

**Layer 2: Protocol-level scaling (time vs num_variables).** Full
`sumcheck()` execution — all v rounds — for each combination of:

- Strategy: time, blendy(k) for several k values
- Shape: multilinear (d=1), inner-product (d=2)
- Field: base only, base→ext3
- Size: v = 16, 18, 20, 22, 24 (65K to 16M evaluations)

Plot time vs 2^v on log-log axes. Time strategy should be linear (slope 1).
Blendy should show a constant-factor gap vs time — the benchmark should
quantify this gap at each k to guide users on the space/time tradeoff.

The key metric is **time per element per round**, not total time. This
normalizes across sizes and makes regressions immediately visible.

**Layer 3: Downstream integration.** Run WHIR's sumcheck bench with our
library as the backend. The metric is **regression detection**: pin a
baseline, alert on >5% regression.

### CI regression tracking

Continuous benchmarks via `github-action-benchmark`:

- **criterion** bench harness with throughput annotations (elements/sec)
- Self-hosted runner on EC2 Sapphire Rapids for stable AVX-512 numbers
- Results committed to `gh-pages` branch with per-benchmark trendlines
- Alert threshold: 10% regression = warning, 15% = CI failure

Benchmark matrix:

```
{time} × {multilinear, IP} × {F64, F64Ext3} × {2^16, 2^20, 2^24}
fold × {F64, F64Ext3} × {2^16, 2^20, 2^24}
```

~20 benchmark points, ~5 minutes on AVX-512 hardware.

## 17. Summary

The design follows Thaler's formalization exactly:

- **One protocol** (Proposition 4.1), parameterized by the polynomial.
- **One trait** (`SumcheckProver`) for the prover's round computation.
- **Three strategies** (time/space/blendy) as construction choices.
- **Three polynomial shapes** (multilinear/product/general) as prover types.
- **One fold** (Lemma 4.3) as the core computational primitive.
- **One verifier** that checks any proof regardless of degree.
- **Partial execution** for protocol composition (GKR, WHIR).
- **Post-state inspection** via `&mut P` ownership for protocol chaining.

The protocol runner, verifier, and fold are the public API. The prover
trait is the extension point. SIMD acceleration is transparent inside
fold. Everything else is internal.
