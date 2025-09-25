use ark_ff::Field;
use ark_std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};
use p3_goldilocks::Goldilocks as P3Goldilocks;
use space_efficient_sumcheck::tests::{
    SmallF128Mont as SmallF128, SmallF64Mont as SmallF64, F128, F64,
};

// // ~128-bit field = Goldilocks^2
// type P3F128 = BinomialExtensionField<P3Goldilocks, 2>;

fn naive_element_wise_mult_ark_bigint_64(c: &mut Criterion) {
    // let's get four vectors and multiply them element-wise
    let len = 2_i64.pow(22);
    let v1: Vec<F64> = (0..len).map(F64::from).collect();
    let v2: Vec<F64> = (0..len).map(F64::from).collect();
    let v3: Vec<F64> = (0..len).map(F64::from).collect();
    let v4: Vec<F64> = (0..len).map(F64::from).collect();
    let mut element_wise_product: Vec<F64> = vec![F64::from(0); len as usize];

    c.bench_function("naive_element_wise_mult_ark_bigint_64", |b| {
        b.iter(|| {
            for i in 0..len as usize {
                element_wise_product[i] = v1.get(i).unwrap()
                    * v2.get(i).unwrap()
                    * v3.get(i).unwrap()
                    * v4.get(i).unwrap();
            }
            black_box(element_wise_product.clone());
        })
    });
}

fn naive_element_wise_mult_ark_small_field_64(c: &mut Criterion) {
    // let's get four vectors and multiply them element-wise
    let len = 2_i64.pow(22);
    let v1: Vec<SmallF64> = (0..len).map(SmallF64::from).collect();
    let v2: Vec<SmallF64> = (0..len).map(SmallF64::from).collect();
    let v3: Vec<SmallF64> = (0..len).map(SmallF64::from).collect();
    let v4: Vec<SmallF64> = (0..len).map(SmallF64::from).collect();
    let mut element_wise_product: Vec<SmallF64> = vec![SmallF64::from(0); len as usize];

    c.bench_function("naive_element_wise_mult_ark_small_field_64", |b| {
        b.iter(|| {
            for i in 0..len as usize {
                element_wise_product[i] = v1.get(i).unwrap()
                    * v2.get(i).unwrap()
                    * v3.get(i).unwrap()
                    * v4.get(i).unwrap();
            }
            black_box(element_wise_product.clone());
        })
    });
}

fn naive_element_wise_mult_p3_64(c: &mut Criterion) {
    // let's get four vectors and multiply them element-wise
    let len = 2_i64.pow(22);
    let v1: Vec<P3Goldilocks> = (0..len)
        .map(|i| P3Goldilocks::from_canonical_u64(i as u64))
        .collect();
    let v2: Vec<P3Goldilocks> = (0..len)
        .map(|i| P3Goldilocks::from_canonical_u64(i as u64))
        .collect();
    let v3: Vec<P3Goldilocks> = (0..len)
        .map(|i| P3Goldilocks::from_canonical_u64(i as u64))
        .collect();
    let v4: Vec<P3Goldilocks> = (0..len)
        .map(|i| P3Goldilocks::from_canonical_u64(i as u64))
        .collect();
    let mut element_wise_product: Vec<P3Goldilocks> =
        vec![P3Goldilocks::from_canonical_u64(0_u64); len as usize];

    c.bench_function("naive_element_wise_mult_p3_64", |b| {
        b.iter(|| {
            for i in 0..len as usize {
                element_wise_product[i] = *v1.get(i).unwrap()
                    * *v2.get(i).unwrap()
                    * *v3.get(i).unwrap()
                    * *v4.get(i).unwrap();
            }
            black_box(element_wise_product.clone());
        })
    });
}

fn naive_element_wise_mult_ark_bigint_128(c: &mut Criterion) {
    // let's get four vectors and multiply them element-wise
    let len = 2_i64.pow(22);
    let v1: Vec<F128> = (0..len).map(F128::from).collect();
    let v2: Vec<F128> = (0..len).map(F128::from).collect();
    let v3: Vec<F128> = (0..len).map(F128::from).collect();
    let v4: Vec<F128> = (0..len).map(F128::from).collect();
    let mut element_wise_product: Vec<F128> = vec![F128::from(0); len as usize];

    c.bench_function("naive_element_wise_mult_ark_bigint", |b| {
        b.iter(|| {
            for i in 0..len as usize {
                element_wise_product[i] = v1.get(i).unwrap()
                    * v2.get(i).unwrap()
                    * v3.get(i).unwrap()
                    * v4.get(i).unwrap();
            }
            black_box(element_wise_product.clone());
        })
    });
}

fn naive_element_wise_mult_ark_small_field_128(c: &mut Criterion) {
    // let's get four vectors and multiply them element-wise
    let len = 2_i64.pow(22);
    let v1: Vec<SmallF128> = (0..len).map(SmallF128::from).collect();
    let v2: Vec<SmallF128> = (0..len).map(SmallF128::from).collect();
    let v3: Vec<SmallF128> = (0..len).map(SmallF128::from).collect();
    let v4: Vec<SmallF128> = (0..len).map(SmallF128::from).collect();
    let mut element_wise_product: Vec<SmallF128> = vec![SmallF128::from(0); len as usize];

    c.bench_function("naive_element_wise_mult_ark_small_field", |b| {
        b.iter(|| {
            for i in 0..len as usize {
                element_wise_product[i] = v1.get(i).unwrap()
                    * v2.get(i).unwrap()
                    * v3.get(i).unwrap()
                    * v4.get(i).unwrap();
            }
            black_box(element_wise_product.clone());
        })
    });
}

// don't include this, we can compare this with ark extension fields in later work
// fn naive_element_wise_mult_p3_binomial_extension_128(c: &mut Criterion) {
//     // let's get four vectors and multiply them element-wise
//     let len = 2_i64.pow(22);
//     let v1: Vec<P3F128> = (0..len)
//         .map(|i| P3F128::from_base(P3Goldilocks::from_canonical_u64(i as u64)))
//         .collect();
//     let v2: Vec<P3F128> = (0..len)
//         .map(|i| P3F128::from_base(P3Goldilocks::from_canonical_u64(i as u64)))
//         .collect();
//     let v3: Vec<P3F128> = (0..len)
//         .map(|i| P3F128::from_base(P3Goldilocks::from_canonical_u64(i as u64)))
//         .collect();
//     let v4: Vec<P3F128> = (0..len)
//         .map(|i| P3F128::from_base(P3Goldilocks::from_canonical_u64(i as u64)))
//         .collect();
//     let mut element_wise_product: Vec<P3F128> =
//         vec![P3F128::from_canonical_u64(0_u64); len as usize];

//     c.bench_function("naive_element_wise_mult_p3_binomial_extension", |b| {
//         b.iter(|| {
//             for i in 0..len as usize {
//                 element_wise_product[i] = *v1.get(i).unwrap()
//                     * *v2.get(i).unwrap()
//                     * *v3.get(i).unwrap()
//                     * *v4.get(i).unwrap();
//             }
//             black_box(element_wise_product.clone());
//         })
//     });
// }

criterion_group!(
    benches,
    naive_element_wise_mult_ark_bigint_64,
    naive_element_wise_mult_ark_small_field_64,
    naive_element_wise_mult_p3_64,
    naive_element_wise_mult_ark_bigint_128,
    naive_element_wise_mult_ark_small_field_128,
);
criterion_main!(benches);
