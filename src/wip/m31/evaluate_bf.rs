use ark_std::{mem, simd::Simd};

#[cfg(feature = "parallel")]
use rayon::{iter::ParallelIterator, prelude::ParallelSlice};

use crate::{
    tests::SmallM31,
    wip::m31::{arithmetic::add::add_v, evaluate_ef::is_serial_better},
};

#[inline(always)]
pub fn add_mod_val<const MODULUS: u32>(a: u32, b: u32) -> u32 {
    let tmp = a + b;
    if tmp >= MODULUS {
        tmp - MODULUS
    } else {
        tmp
    }
}

#[inline(always)]
pub fn mul_mod_val<const MODULUS: u32>(a: u32, b: u32) -> u32 {
    let res = SmallM31::from(a) * SmallM31::from(b);
    unsafe { mem::transmute::<SmallM31, u32>(res) }
}

#[inline(always)]
fn sum_indexwise<const MODULUS: u32>(
    a1: [u32; 4],
    a2: [u32; 4],
    a3: [u32; 4],
    a4: [u32; 4],
) -> (u32, u32) {
    let mut even_sum = 0u32;
    let mut odd_sum = 0u32;

    let add_mod = |acc: &mut u32, x: u32| {
        let tmp = *acc + x;
        *acc = if tmp >= MODULUS { tmp - MODULUS } else { tmp };
    };

    for arr in [&a1, &a2, &a3, &a4] {
        add_mod(&mut even_sum, arr[0]);
        add_mod(&mut even_sum, arr[2]);

        add_mod(&mut odd_sum, arr[1]);
        add_mod(&mut odd_sum, arr[3]);
    }

    (even_sum, odd_sum)
}

fn reduce_sum_packed_bf<const MODULUS: u32>(src: &[u32]) -> (u32, u32) {
    // Must be list of Fp4SmallM31s in memory like: &[a0, a1, a2, a3, b0, b1, b2, b3]
    // and they must come in pairs of two
    assert!(src.len().is_multiple_of(8));

    let modulus = Simd::<u32, 4>::splat(MODULUS);
    let mut acc0: Simd<u32, 4> = Simd::<u32, 4>::splat(0);
    let mut acc1: Simd<u32, 4> = Simd::<u32, 4>::splat(0);
    let mut acc2: Simd<u32, 4> = Simd::<u32, 4>::splat(0);
    let mut acc3: Simd<u32, 4> = Simd::<u32, 4>::splat(0);

    let len = src.len();
    let chunk_size = 4 * 4;
    for i in (0..len).step_by(chunk_size) {
        acc0 = add_v(&acc0, &Simd::<u32, 4>::from_slice(&src[i..i + 4]), &modulus);
        acc1 = add_v(
            &acc1,
            &Simd::<u32, 4>::from_slice(&src[i + 4..i + 2 * 4]),
            &modulus,
        );
        acc2 = add_v(
            &acc2,
            &Simd::<u32, 4>::from_slice(&src[i + 2 * 4..i + 3 * 4]),
            &modulus,
        );
        acc3 = add_v(
            &acc3,
            &Simd::<u32, 4>::from_slice(&src[i + 3 * 4..i + 4 * 4]),
            &modulus,
        );
    }

    sum_indexwise::<MODULUS>(
        acc0.to_array(),
        acc1.to_array(),
        acc2.to_array(),
        acc3.to_array(),
    )
}

pub fn evaluate_bf<const MODULUS: u32>(src: &[SmallM31]) -> (SmallM31, SmallM31) {
    // TODO (z-tech): break even is machine dependent
    if is_serial_better(src.len(), 1 << 16) || !cfg!(feature = "parallel") {
        let src_raw: &[u32] =
            unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u32, src.len()) };
        let (sum0_raw, sum1_raw) = reduce_sum_packed_bf::<MODULUS>(src_raw);
        return (
            unsafe { mem::transmute::<u32, SmallM31>(sum0_raw) },
            unsafe { mem::transmute::<u32, SmallM31>(sum1_raw) },
        );
    }

    #[cfg(feature = "parallel")]
    {
        let n_threads = rayon::current_num_threads();
        let chunk_size = src.len() / n_threads;
        let sums = src
            .par_chunks(chunk_size) // minimum block = SIMD width
            .map(|chunk| {
                let chunk_raw: &[u32] = unsafe {
                    core::slice::from_raw_parts(chunk.as_ptr() as *const u32, chunk.len())
                };
                reduce_sum_packed_bf::<MODULUS>(chunk_raw)
            })
            .reduce(
                || (0, 0),
                |(e1, o1), (e2, o2)| {
                    (
                        add_mod_val::<MODULUS>(e1, e2),
                        add_mod_val::<MODULUS>(o1, o2),
                    )
                },
            );
        let (sum_0_u32, sum_1_u32) = sums;
        let sum_0: SmallM31 = unsafe { mem::transmute::<u32, SmallM31>(sum_0_u32) };
        let sum_1: SmallM31 = unsafe { mem::transmute::<u32, SmallM31>(sum_1_u32) };
        (sum_0, sum_1)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use crate::multilinear::pairwise;
    use crate::tests::SmallM31;
    use crate::wip::m31::evaluate_bf::evaluate_bf;

    #[test]
    fn sanity() {
        // setup
        const LEN: usize = 1 << 20;
        let mut rng = test_rng();

        // random elements
        let src: Vec<SmallM31> = (0..LEN).map(|_| SmallM31::rand(&mut rng)).collect();

        // run function
        let expected = pairwise::evaluate(&src);
        let received = evaluate_bf::<2_147_483_647>(&src);

        assert_eq!(expected, received);
    }
}
