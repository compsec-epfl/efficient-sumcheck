# Non-arkworks field support

By default, the library provides a blanket `SumcheckField` implementation for
all `ark_ff::Field` types. Non-arkworks users can compile with
`--no-default-features` and implement the trait for their own field type.

## `SumcheckField` trait

```rust
pub trait SumcheckField:
    Copy + Send + Sync + PartialEq + Debug
    + Add + Sub + Mul + Neg + AddAssign + SubAssign + MulAssign
    + Sum + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn from_u64(val: u64) -> Self;
    fn inverse(&self) -> Option<Self>;
}
```

## SIMD opt-in via `SimdRepr`

To enable SIMD acceleration for a non-arkworks Goldilocks type, implement
`SimdRepr`. The `zerocopy` bounds (`IntoBytes + FromBytes + Immutable`) provide
compile-time layout verification — no `unsafe` needed from the implementor.

```rust
#[derive(Clone, Copy, Debug, PartialEq,
         zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
#[repr(transparent)]
struct MyGoldilocks(u64);

impl SumcheckField for MyGoldilocks {
    const ZERO: Self = MyGoldilocks(0);
    const ONE: Self = MyGoldilocks(1); // Montgomery form of 1
    fn from_u64(val: u64) -> Self { /* ... */ }
    fn inverse(&self) -> Option<Self> { /* ... */ }
    fn _simd_field_config() -> Option<SimdFieldConfig> {
        Some(SimdFieldConfig { modulus: GOLDILOCKS_P, element_bytes: 8 })
    }
}

impl SimdRepr for MyGoldilocks {
    fn modulus() -> u64 { GOLDILOCKS_P }
}
```

Extension fields work the same way:

```rust
#[derive(Clone, Copy, Debug, PartialEq,
         zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
#[repr(transparent)]
struct MyExt3([u64; 3]);

impl SumcheckField for MyExt3 {
    fn extension_degree() -> u64 { 3 }
    fn _simd_field_config() -> Option<SimdFieldConfig> {
        Some(SimdFieldConfig { modulus: GOLDILOCKS_P, element_bytes: 8 })
    }
    // ...
}

impl SimdRepr for MyExt3 {
    fn modulus() -> u64 { GOLDILOCKS_P }
}
```

## Feature flags

```toml
[features]
default = ["arkworks", "parallel"]
arkworks = ["ark-ff", "ark-poly", "ark-serialize", "ark-std", "spongefish"]
parallel = ["rayon"]
```

- `arkworks` (default): blanket `SumcheckField` impl for `ark_ff::Field`
- `parallel` (default): rayon parallelism for fold and round computation
- `--no-default-features`: pure `SumcheckField` library, no arkworks dependency
