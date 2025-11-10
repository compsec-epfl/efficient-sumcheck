#![feature(portable_simd)]

use ark_ff::{PrimeField, SmallFp};
use ark_std::{
    simd::{cmp::SimdPartialOrd, u16x4, Mask, Simd, num::SimdUint, num::SimdInt},
    test_rng, hint::black_box, UniformRand,
};

use criterion::{criterion_group, criterion_main, Criterion};
use efficient_sumcheck::tests::{SmallGoldilocks, SmallM31, F64 as Goldilocks, M31, SmallF16};

const F16_MODULUS: u16 = 65521;
const MONT_P_DASH: u16 = 0xEEEF;

#[inline(always)]
fn montgomery_mul_u16x4(a: Simd<u16, LANES>, b: Simd<u16, LANES>) -> Simd<u16, LANES> {
    // widen to u32 lanes for intermediate math
    let a32: Simd<u32, LANES> = a.cast();
    let b32: Simd<u32, LANES> = b.cast();

    let p32 = Simd::<u32, LANES>::splat(F16_MODULUS as u32);
    let p_dash32 = Simd::<u32, LANES>::splat(MONT_P_DASH as u32);
    let rmask32 = Simd::<u32, LANES>::splat(0xFFFF);

    // T = A * B (mod 2^32 via u32 wrapping)
    let t = a32 * b32;

    // m = (T * n') mod R  (only low 16 bits matter)
    let m = (t * p_dash32) & rmask32;

    // sum = T + m * p (mod 2^32)
    let mp = m * p32;
    let sum = t + mp;

    // carry = 1 if overflow in sum, else 0
    let carry_mask = sum.simd_lt(t); // if wrapped, sum < T
    let carry_ones: Simd<u32, LANES> = carry_mask.to_int().cast();
    let carry = carry_ones & Simd::<u32, LANES>::splat(1);

    // u = (sum >> 16) + (carry << 16)
    let hi = sum >> Simd::<u32, LANES>::splat(16);
    let carry_hi = carry << Simd::<u32, LANES>::splat(16);
    let mut u = hi + carry_hi;

    // if u >= p, subtract p (per lane)
    let ge_mask = u.simd_ge(p32);
    let u_minus_p = u - p32;
    u = ge_mask.select(u_minus_p, u);

    // back to u16 Montgomery repr
    u.cast()
}
// fn montgomery_mul_u16x4(a: Simd<u16, LANES>, b: Simd<u16, LANES>) -> Simd<u16, LANES> {
//     // widen to u32 so intermediate fits (32-bit is plenty)
//     let a32: Simd<u32, LANES> = a.cast();
//     let b32: Simd<u32, LANES> = b.cast();

//     let p32 = Simd::<u32, LANES>::splat(F16_MODULUS as u32);
//     let p_dash32 = Simd::<u32, LANES>::splat(MONT_P_DASH as u32);
//     let rmask32 = Simd::<u32, LANES>::splat(0xFFFF);

//     // T = A * B  (mod 2^32 via normal wrapping semantics)
//     let t = a32 * b32;

//     // m = (T * p') mod R, just keep low 16 bits
//     let m = (t * p_dash32) & rmask32;

//     // u = (T + m*p) / R  (>> 16)
//     let u = (t + m * p32) >> 16;

//     // if u >= p, subtract p (per lane)
//     let ge_mask = u.simd_ge(p32);          // Mask<LANES>
//     let u_minus_p = u - p32;
//     let u_red = ge_mask.select(u_minus_p, u);

//     // back to u16 Montgomery repr
//     u_red.cast()
// }

// TODO (z-tech): this is the benchmark we should hit with both Neon and AVX
const LANES: usize = 4;
pub fn mul_assign_16_bit_vectorized(a: &mut [u16], b: &[u16]) {
    // u16 modulus vector
    let modulus_16: Simd<u16, LANES> = Simd::splat(F16_MODULUS);
    let modulus_32: Simd<u32, LANES> = modulus_16.cast();
    for i in (0..a.len()).step_by(16) {
        let a_1: Simd<u16, LANES> = u16x4::from_slice(&a[i..i + 4]);
        // println!("a: {:?}", a_1);
        let a_2: Simd<u16, LANES> = u16x4::from_slice(&a[i + 4..i + 8]);
        let a_3: Simd<u16, LANES> = u16x4::from_slice(&a[i + 8..i + 12]);
        let a_4: Simd<u16, LANES> = u16x4::from_slice(&a[i + 12..i + 16]);

        let b_1: Simd<u16, LANES> = u16x4::from_slice(&b[i..i + 4]);
        let b_2: Simd<u16, LANES> = u16x4::from_slice(&b[i + 4..i + 8]);
        let b_3: Simd<u16, LANES> = u16x4::from_slice(&b[i + 8..i + 12]);
        let b_4: Simd<u16, LANES> = u16x4::from_slice(&b[i + 12..i + 16]);

        let a_1_reduced = montgomery_mul_u16x4(a_1, b_1);
        let a_2_reduced = montgomery_mul_u16x4(a_2, b_2);
        let a_3_reduced = montgomery_mul_u16x4(a_3, b_3);
        let a_4_reduced = montgomery_mul_u16x4(a_4, b_4);

        // let mut a_1_u32: Simd<u32, LANES> = a_1.cast();
        // let mut a_2_u32: Simd<u32, LANES> = a_2.cast();
        // let mut a_3_u32: Simd<u32, LANES> = a_3.cast();
        // let mut a_4_u32: Simd<u32, LANES> = a_4.cast();

        // let b_1_u32: Simd<u32, LANES> = b_1.cast();
        // let b_2_u32: Simd<u32, LANES> = b_2.cast();
        // let b_3_u32: Simd<u32, LANES> = b_3.cast();
        // let b_4_u32: Simd<u32, LANES> = b_4.cast();

        // a_1_u32 *= b_1_u32;
        // a_1_u32 %= modulus_32;

        // a_2_u32 *= b_2_u32;
        // a_2_u32 %= modulus_32;

        // a_3_u32 *= b_3_u32;
        // a_3_u32 %= modulus_32;

        // a_4_u32 *= b_4_u32;
        // a_4_u32 %= modulus_32;

        // let a_1_reduced: Simd<u16, LANES> = a_1_u32.cast();
        // let a_2_reduced: Simd<u16, LANES> = a_2_u32.cast();
        // let a_3_reduced: Simd<u16, LANES> = a_3_u32.cast();
        // let a_4_reduced: Simd<u16, LANES> = a_4_u32.cast();

        a[i..i + 4].copy_from_slice(a_1_reduced.as_array());
        a[i + 4..i + 8].copy_from_slice(a_2_reduced.as_array());
        a[i + 8..i + 12].copy_from_slice(a_3_reduced.as_array());
        a[i + 12..i + 16].copy_from_slice(a_4_reduced.as_array());
    }
}

fn bench_mul_16_bit_prime(c: &mut Criterion) { 
    let mut rng = test_rng();
    let mut a: Vec<SmallF16> = (0..1<<20).map(|_| SmallF16::rand(&mut rng)).collect();
    let b: Vec<SmallF16> = (0..1<<20).map(|_| SmallF16::rand(&mut rng)).collect();

    c.bench_function("pairwise_mul_16_bit_prime", |bencher| {
        bencher.iter(|| {
            for i in 0..a.len() {
                a[i] *= &b[i];
            }
            black_box(&a);
        });
    });
}

fn bench_mul_16_bit_prime_vectorized(c: &mut Criterion) { 
    let mut rng = test_rng();
    let mut a: Vec<SmallF16> = (0..1<<20).map(|_| SmallF16::rand(&mut rng)).collect();
    let b: Vec<SmallF16> =  (0..1<<20).map(|_| SmallF16::rand(&mut rng)).collect();

    let a_u16: &mut [u16] = unsafe {
        core::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u16, a.len())
    };
    let b_u16: &[u16] = unsafe {
        core::slice::from_raw_parts(b.as_ptr() as *const u16, b.len())
    };

    c.bench_function("pairwise_mul_16_bit_prime_vectorized", |bencher| {
        bencher.iter(|| {
            mul_assign_16_bit_vectorized(a_u16, b_u16);
            black_box(&a);
        });
    });
}

criterion_group!(
    benches,
    bench_mul_16_bit_prime,
    bench_mul_16_bit_prime_vectorized,
);
criterion_main!(benches);
