use ark_ff::{
    AdditiveGroup, BigInt, FftField, Field, LegendreSymbol, One, PrimeField, SqrtPrecomputation,
    Zero,
};
use ark_serialize::{
    buffer_byte_size, CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, Compress, EmptyFlags, Flags, SerializationError, Valid, Validate,
};
use ark_std::{
    cmp::*,
    fmt::{Display, Formatter, Result as FmtResult},
    hash::Hash,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
    string::*,
};
use core::iter;
use educe::Educe;
use itertools::Itertools;
use num_traits::Unsigned;

/// A trait that specifies the configuration of a prime field.
/// Also specifies how to perform arithmetic on field elements.
pub trait SmallFpConfig: Send + Sync + 'static + Sized {
    type T: Copy
        + Default
        + PartialEq
        + Eq
        + Hash
        + Sync
        + Send
        + PartialOrd
        + Display
        + Unsigned
        + std::fmt::Debug
        + std::ops::Add<Output = Self::T>
        + std::ops::Sub<Output = Self::T>
        + std::ops::Mul<Output = Self::T>
        + std::ops::Div<Output = Self::T>
        + std::ops::Rem<Output = Self::T>
        + Into<u128>
        + TryFrom<u128>;

    /// The modulus of the field.
    const MODULUS: Self::T;
    const MODULUS_128: u128;

    /// A multiplicative generator of the field.
    /// `Self::GENERATOR` is an element having multiplicative order
    /// `Self::MODULUS - 1`.
    const GENERATOR: SmallFp<Self>;

    /// Additive identity of the field, i.e. the element `e`
    /// such that, for all elements `f` of the field, `e + f = f`.
    const ZERO: SmallFp<Self>;

    /// Multiplicative identity of the field, i.e. the element `e`
    /// such that, for all elements `f` of the field, `e * f = f`.
    const ONE: SmallFp<Self>;

    /// Let `N` be the size of the multiplicative group defined by the field.
    /// Then `TWO_ADICITY` is the two-adicity of `N`, i.e. the integer `s`
    /// such that `N = 2^s * t` for some odd integer `t`.
    const TWO_ADICITY: u32;

    /// 2^s root of unity computed by GENERATOR^t
    const TWO_ADIC_ROOT_OF_UNITY: SmallFp<Self>;

    /// An integer `b` such that there exists a multiplicative subgroup
    /// of size `b^k` for some integer `k`.
    const SMALL_SUBGROUP_BASE: Option<u32> = None;

    /// The integer `k` such that there exists a multiplicative subgroup
    /// of size `Self::SMALL_SUBGROUP_BASE^k`.
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = None;

    /// GENERATOR^((MODULUS-1) / (2^s *
    /// SMALL_SUBGROUP_BASE^SMALL_SUBGROUP_BASE_ADICITY)) Used for mixed-radix
    /// FFT.
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<SmallFp<Self>> = None;

    /// Precomputed material for use when computing square roots.
    /// Currently uses the generic Tonelli-Shanks,
    /// which works for every modulus.
    const SQRT_PRECOMP: Option<SqrtPrecomputation<SmallFp<Self>>>;

    /// Set a += b.
    fn add_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>);

    /// Set a -= b.
    fn sub_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>);

    /// Set a = a + a.
    fn double_in_place(a: &mut SmallFp<Self>);

    /// Set a = -a;
    fn neg_in_place(a: &mut SmallFp<Self>);

    /// Set a *= b.
    fn mul_assign(a: &mut SmallFp<Self>, b: &SmallFp<Self>);

    /// Compute the inner product `<a, b>`.
    fn sum_of_products<const T: usize>(
        a: &[SmallFp<Self>; T],
        b: &[SmallFp<Self>; T],
    ) -> SmallFp<Self>;

    /// Set a *= a.
    fn square_in_place(a: &mut SmallFp<Self>);

    /// Compute a^{-1} if `a` is not zero.
    fn inverse(a: &SmallFp<Self>) -> Option<SmallFp<Self>>;

    /// Construct a field element from an integer in the range
    /// `0..(Self::MODULUS - 1)`. Returns `None` if the integer is outside
    /// this range.
    fn from_bigint(other: BigInt<2>) -> Option<SmallFp<Self>>;

    /// Convert a field element to an integer in the range `0..(Self::MODULUS -
    /// 1)`.
    fn into_bigint(other: SmallFp<Self>) -> BigInt<2>;
}

/// Represents an element of the prime field F_p, where `p == P::MODULUS`.
/// This type can represent elements in any field of size at most N * 64 bits.
#[derive(Educe)]
#[educe(Default, Hash, Clone, Copy, PartialEq, Eq)]
pub struct SmallFp<P: SmallFpConfig> {
    pub value: P::T,
    _phantom: PhantomData<P>,
}

impl<P: SmallFpConfig> SmallFp<P> {
    #[doc(hidden)]
    #[inline]
    pub fn is_geq_modulus(&self) -> bool {
        self.value >= P::MODULUS
    }

    #[inline]
    fn subtract_modulus(&mut self) {
        self.value = self.value.sub(P::MODULUS);
    }

    pub const fn new(value: P::T) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
    // TODO: Constant 2 is here becuase we fixed BigInt<2> this will be generic in the future
    fn num_bits_to_shave() -> usize {
        64 * 2 - (Self::MODULUS_BIT_SIZE as usize)
    }
}

impl<P: SmallFpConfig> ark_std::fmt::Debug for SmallFp<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> ark_std::fmt::Result {
        ark_std::fmt::Debug::fmt(&self.into_bigint(), f)
    }
}

impl<P: SmallFpConfig> Zero for SmallFp<P> {
    #[inline]
    fn zero() -> Self {
        P::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == P::ZERO
    }
}

impl<P: SmallFpConfig> One for SmallFp<P> {
    #[inline]
    fn one() -> Self {
        P::ONE
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == P::ONE
    }
}

impl<P: SmallFpConfig> AdditiveGroup for SmallFp<P> {
    type Scalar = Self;
    const ZERO: Self = P::ZERO;

    #[inline]
    fn double(&self) -> Self {
        let mut temp = *self;
        AdditiveGroup::double_in_place(&mut temp);
        temp
    }

    #[inline]
    fn double_in_place(&mut self) -> &mut Self {
        P::double_in_place(self);
        self
    }

    #[inline]
    fn neg_in_place(&mut self) -> &mut Self {
        P::neg_in_place(self);
        self
    }
}

impl<P: SmallFpConfig> Field for SmallFp<P> {
    type BasePrimeField = Self;
    // type BasePrimeFieldIter = std::iter::Once<Self::BasePrimeField>;

    const SQRT_PRECOMP: Option<SqrtPrecomputation<Self>> = P::SQRT_PRECOMP;
    const ONE: Self = P::ONE;
    // const ZERO: Self = P::ZERO;

    fn extension_degree() -> u64 {
        1
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        elem
    }

    fn to_base_prime_field_elements(
        &self,
    ) -> impl Iterator<Item = <Self as ark_ff::Field>::BasePrimeField> {
        iter::once(*self)
    }

    fn from_base_prime_field_elems(
        elems: impl IntoIterator<Item = Self::BasePrimeField>,
    ) -> Option<Self> {
        todo!()
    }

    // fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
    //     elems.into_iter().exactly_one().ok().copied()
    // }

    #[inline]
    fn characteristic() -> &'static [u64] {
        // if P::MODULUS <= u64::MAX {
        //     &[P::MODULUS]
        // } else {
        //     &[0]
        // }
        &[0]
    }

    #[inline]
    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        P::sum_of_products(a, b)
    }

    #[inline]
    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        if F::BIT_SIZE > 8 {
            None
        } else {
            let shave_bits = Self::num_bits_to_shave();
            let mut result_bytes: super::const_helpers::SerBuffer<2> =
                crate::fields::const_helpers::SerBuffer::zeroed();
            // Copy the input into a temporary buffer.
            result_bytes.copy_from_u8_slice(bytes);
            // This mask retains everything in the last limb
            // that is below `P::MODULUS_BIT_SIZE`.
            let last_limb_mask =
                (u64::MAX.checked_shr(shave_bits as u32).unwrap_or(0)).to_le_bytes();
            let mut last_bytes_mask = [0u8; 9];
            last_bytes_mask[..8].copy_from_slice(&last_limb_mask);

            // Length of the buffer containing the field element and the flag.
            let output_byte_size = buffer_byte_size(Self::MODULUS_BIT_SIZE as usize + F::BIT_SIZE);
            // Location of the flag is the last byte of the serialized
            // form of the field element.
            let flag_location = output_byte_size - 1;

            // At which byte is the flag located in the last limb?
            // TODO: Constant 2 is here becuase we fixed BigInt<2> this will be generic in the future
            let flag_location_in_last_limb = flag_location.saturating_sub(8 * (2 - 1));

            // Take all but the last 9 bytes.
            let last_bytes = result_bytes.last_n_plus_1_bytes_mut();

            // The mask only has the last `F::BIT_SIZE` bits set
            let flags_mask = u8::MAX.checked_shl(8 - (F::BIT_SIZE as u32)).unwrap_or(0);

            // Mask away the remaining bytes, and try to reconstruct the
            // flag
            let mut flags: u8 = 0;
            for (i, (b, m)) in last_bytes.zip(&last_bytes_mask).enumerate() {
                if i == flag_location_in_last_limb {
                    flags = *b & flags_mask
                }
                *b &= m;
            }
            // TODO: Constant 2 is here becuase we fixed BigInt<2> this will be generic in the future
            Self::deserialize_compressed(&result_bytes.as_slice()[..(2 * 8)])
                .ok()
                .and_then(|f| F::from_u8(flags).map(|flag| (f, flag)))
        }
    }

    #[inline]
    fn square(&self) -> Self {
        let mut temp = *self;
        temp.square_in_place();
        temp
    }

    fn square_in_place(&mut self) -> &mut Self {
        P::square_in_place(self);
        self
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        P::inverse(self)
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        self.inverse().map(|inverse| {
            *self = inverse;
            self
        })
    }

    /// The Frobenius map has no effect in a prime field.
    #[inline]
    fn frobenius_map_in_place(&mut self, _: usize) {}

    #[inline]
    fn legendre(&self) -> LegendreSymbol {
        // s = self^((MODULUS - 1) // 2)
        let s = self.pow(Self::MODULUS_MINUS_ONE_DIV_TWO);
        if s.is_zero() {
            LegendreSymbol::Zero
        } else if s.is_one() {
            LegendreSymbol::QuadraticResidue
        } else {
            LegendreSymbol::QuadraticNonResidue
        }
    }

    fn mul_by_base_prime_field(&self, elem: &Self::BasePrimeField) -> Self {
        todo!()
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
    }

    fn sqrt(&self) -> Option<Self> {
        match Self::SQRT_PRECOMP {
            Some(tv) => tv.sqrt(self),
            None => std::unimplemented!(),
        }
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        (*self).sqrt().map(|sqrt| {
            *self = sqrt;
            self
        })
    }

    fn frobenius_map(&self, power: usize) -> Self {
        let mut this = *self;
        this.frobenius_map_in_place(power);
        this
    }

    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();

        for i in ark_ff::BitIteratorBE::without_leading_zeros(exp) {
            res.square_in_place();

            if i {
                res *= self;
            }
        }
        res
    }

    fn pow_with_table<S: AsRef<[u64]>>(powers_of_2: &[Self], exp: S) -> Option<Self> {
        let mut res = Self::one();
        for (pow, bit) in ark_ff::BitIteratorLE::without_trailing_zeros(exp).enumerate() {
            if bit {
                res *= powers_of_2.get(pow)?;
            }
        }
        Some(res)
    }
}

const fn const_to_bigint(value: u128) -> BigInt<2> {
    let low = (value & 0xFFFFFFFFFFFFFFFF) as u64;
    let high = (value >> 64) as u64;
    BigInt::<2>::new([high, low])
}

// TODO: Make this generic for BigInt<N>
impl<P: SmallFpConfig> PrimeField for SmallFp<P> {
    type BigInt = BigInt<2>;

    const MODULUS: Self::BigInt = const_to_bigint(P::MODULUS_128);
    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt = Self::MODULUS.divide_by_2_round_down();
    const MODULUS_BIT_SIZE: u32 = Self::MODULUS.const_num_bits();
    const TRACE: Self::BigInt = Self::MODULUS.two_adic_coefficient();
    const TRACE_MINUS_ONE_DIV_TWO: Self::BigInt = Self::TRACE.divide_by_2_round_down();

    #[inline]
    fn from_bigint(r: BigInt<2>) -> Option<Self> {
        P::from_bigint(r)
    }

    fn into_bigint(self) -> BigInt<2> {
        P::into_bigint(self)
    }
}

impl<P: SmallFpConfig> FftField for SmallFp<P> {
    const GENERATOR: Self = P::GENERATOR;
    const TWO_ADICITY: u32 = P::TWO_ADICITY;
    const TWO_ADIC_ROOT_OF_UNITY: Self = P::TWO_ADIC_ROOT_OF_UNITY;
    const SMALL_SUBGROUP_BASE: Option<u32> = P::SMALL_SUBGROUP_BASE;
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = P::SMALL_SUBGROUP_BASE_ADICITY;
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Self> = P::LARGE_SUBGROUP_ROOT_OF_UNITY;
}

/// Note that this implementation of `Ord` compares field elements viewing
/// them as integers in the range 0, 1, ..., P::MODULUS - 1. However, other
/// implementations of `PrimeField` might choose a different ordering, and
/// as such, users should use this `Ord` for applications where
/// any ordering suffices (like in a BTreeMap), and not in applications
/// where a particular ordering is required.
impl<P: SmallFpConfig> Ord for SmallFp<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_bigint().cmp(&other.into_bigint())
    }
}

/// Note that this implementation of `PartialOrd` compares field elements
/// viewing them as integers in the range 0, 1, ..., `P::MODULUS` - 1. However,
/// other implementations of `PrimeField` might choose a different ordering, and
/// as such, users should use this `PartialOrd` for applications where
/// any ordering suffices (like in a BTreeMap), and not in applications
/// where a particular ordering is required.
impl<P: SmallFpConfig> PartialOrd for SmallFp<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: SmallFpConfig> From<u128> for SmallFp<P> {
    fn from(other: u128) -> Self {
        let other_as_t = match P::T::try_from(other.into()) {
            Ok(val) => val,
            Err(_) => {
                let modulus_as_u128: u128 = P::MODULUS.into();
                let reduced = other % modulus_as_u128;
                P::T::try_from(reduced).unwrap_or_else(|_| panic!("Reduced value should fit in T"))
            }
        };
        let val = other_as_t % P::MODULUS;
        SmallFp::new(val)
    }
}

impl<P: SmallFpConfig> From<i128> for SmallFp<P> {
    fn from(other: i128) -> Self {
        let abs = other.unsigned_abs().into();
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: SmallFpConfig> From<bool> for SmallFp<P> {
    fn from(other: bool) -> Self {
        if other == true {
            P::ONE
        } else {
            P::ZERO
        }
    }
}

impl<P: SmallFpConfig> From<u64> for SmallFp<P> {
    fn from(other: u64) -> Self {
        let other_as_t = match P::T::try_from(other.into()) {
            Ok(val) => val,
            Err(_) => {
                let modulus_as_u128: u128 = P::MODULUS.into();
                let reduced = (other as u128) % modulus_as_u128;
                P::T::try_from(reduced).unwrap_or_else(|_| panic!("Reduced value should fit in T"))
            }
        };
        let val = other_as_t % P::MODULUS;
        SmallFp::new(val)
    }
}

impl<P: SmallFpConfig> From<i64> for SmallFp<P> {
    fn from(other: i64) -> Self {
        let abs = other.unsigned_abs().into();
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: SmallFpConfig> From<u32> for SmallFp<P> {
    fn from(other: u32) -> Self {
        let other_as_t = match P::T::try_from(other.into()) {
            Ok(val) => val,
            Err(_) => {
                let modulus_as_u128: u128 = P::MODULUS.into();
                let reduced = (other as u128) % modulus_as_u128;
                P::T::try_from(reduced).unwrap_or_else(|_| panic!("Reduced value should fit in T"))
            }
        };
        let val = other_as_t % P::MODULUS;
        SmallFp::new(val)
    }
}

impl<P: SmallFpConfig> From<i32> for SmallFp<P> {
    fn from(other: i32) -> Self {
        let abs = other.unsigned_abs().into();
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: SmallFpConfig> From<u16> for SmallFp<P> {
    fn from(other: u16) -> Self {
        let other_as_t = match P::T::try_from(other.into()) {
            Ok(val) => val,
            Err(_) => {
                let modulus_as_u128: u128 = P::MODULUS.into();
                let reduced = (other as u128) % modulus_as_u128;
                P::T::try_from(reduced).unwrap_or_else(|_| panic!("Reduced value should fit in T"))
            }
        };
        let val = other_as_t % P::MODULUS;
        SmallFp::new(val)
    }
}

impl<P: SmallFpConfig> From<i16> for SmallFp<P> {
    fn from(other: i16) -> Self {
        let abs = other.unsigned_abs().into();
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: SmallFpConfig> From<u8> for SmallFp<P> {
    fn from(other: u8) -> Self {
        let other_as_t = match P::T::try_from(other.into()) {
            Ok(val) => val,
            Err(_) => {
                let modulus_as_u128: u128 = P::MODULUS.into();
                let reduced = (other as u128) % modulus_as_u128;
                P::T::try_from(reduced).unwrap_or_else(|_| panic!("Reduced value should fit in T"))
            }
        };
        let val = other_as_t % P::MODULUS;
        SmallFp::new(val)
    }
}

impl<P: SmallFpConfig> From<i8> for SmallFp<P> {
    fn from(other: i8) -> Self {
        let abs = other.unsigned_abs().into();
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: SmallFpConfig> ark_std::rand::distributions::Distribution<SmallFp<P>>
    for ark_std::rand::distributions::Standard
{
    #[inline]
    // TODO: fix this
    fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> SmallFp<P> {
        loop {
            let tmp = SmallFp::from(1);

            return tmp;
        }
    }
}

impl<P: SmallFpConfig> CanonicalSerializeWithFlags for SmallFp<P> {
    fn serialize_with_flags<W: ark_std::io::Write, F: Flags>(
        &self,
        writer: W,
        flags: F,
    ) -> Result<(), SerializationError> {
        // All reasonable `Flags` should be less than 8 bits in size
        // (256 values are enough for anyone!)
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }

        // Calculate the number of bytes required to represent a field element
        // serialized with `flags`. If `F::BIT_SIZE < 8`,
        // this is at most `N * 8 + 1`
        let output_byte_size = buffer_byte_size(Self::MODULUS_BIT_SIZE as usize + F::BIT_SIZE);

        // Write out `self` to a temporary buffer.
        // The size of the buffer is $byte_size + 1 because `F::BIT_SIZE`
        // is at most 8 bits.
        let mut bytes = crate::fields::const_helpers::SerBuffer::zeroed();
        bytes.copy_from_u64_slice(&self.into_bigint().0);
        // Mask out the bits of the last byte that correspond to the flag.
        bytes[output_byte_size - 1] |= flags.u8_bitmask();

        bytes.write_up_to(writer, output_byte_size)?;
        Ok(())
    }

    // Let `m = 8 * n` for some `n` be the smallest multiple of 8 greater
    // than `P::MODULUS_BIT_SIZE`.
    // If `(m - P::MODULUS_BIT_SIZE) >= F::BIT_SIZE` , then this method returns `n`;
    // otherwise, it returns `n + 1`.
    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        buffer_byte_size(Self::MODULUS_BIT_SIZE as usize + F::BIT_SIZE)
    }
}

impl<P: SmallFpConfig> CanonicalSerialize for SmallFp<P> {
    #[inline]
    fn serialize_with_mode<W: ark_std::io::Write>(
        &self,
        writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.serialize_with_flags(writer, EmptyFlags)
    }

    #[inline]
    fn serialized_size(&self, _compress: Compress) -> usize {
        self.serialized_size_with_flags::<EmptyFlags>()
    }
}

impl<P: SmallFpConfig> CanonicalDeserializeWithFlags for SmallFp<P> {
    fn deserialize_with_flags<R: ark_std::io::Read, F: Flags>(
        reader: R,
    ) -> Result<(Self, F), SerializationError> {
        // All reasonable `Flags` should be less than 8 bits in size
        // (256 values are enough for anyone!)
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        // Calculate the number of bytes required to represent a field element
        // serialized with `flags`.
        let output_byte_size = Self::zero().serialized_size_with_flags::<F>();

        let mut masked_bytes = crate::fields::const_helpers::SerBuffer::zeroed();
        masked_bytes.read_exact_up_to(reader, output_byte_size)?;
        let flags = F::from_u8_remove_flags(&mut masked_bytes[output_byte_size - 1])
            .ok_or(SerializationError::UnexpectedFlags)?;

        let self_integer = masked_bytes.to_bigint();
        Self::from_bigint(self_integer)
            .map(|v| (v, flags))
            .ok_or(SerializationError::InvalidData)
    }
}

impl<P: SmallFpConfig> Valid for SmallFp<P> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<P: SmallFpConfig> CanonicalDeserialize for SmallFp<P> {
    fn deserialize_with_mode<R: ark_std::io::Read>(
        reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(r, _)| r)
    }
}

pub enum ParseSmallFpError {
    Empty,
    InvalidFormat,
    InvalidLeadingZero,
}

impl<P: SmallFpConfig> FromStr for SmallFp<P> {
    type Err = ParseSmallFpError;

    /// Interpret a string of numbers as a (congruent) prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(ParseSmallFpError::Empty);
        }
        if s.starts_with('0') && s.len() > 1 {
            return Err(ParseSmallFpError::InvalidLeadingZero);
        }

        match s.parse::<u128>() {
            // TODO: This should not be u128 but P::T
            Ok(val) => Ok(SmallFp::from(val)),
            Err(_) => Err(ParseSmallFpError::InvalidFormat),
        }
    }
}

/// Outputs a string containing the value of `self`,
/// represented as a decimal without leading zeroes.
impl<P: SmallFpConfig> Display for SmallFp<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let string = self.value.to_string();
        write!(f, "{}", string)
    }
}

impl<P: SmallFpConfig> Neg for SmallFp<P> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        P::neg_in_place(&mut self);
        self
    }
}

impl<P: SmallFpConfig> Add<&SmallFp<P>> for SmallFp<P> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: &Self) -> Self {
        self += other;
        self
    }
}

impl<P: SmallFpConfig> Sub<&SmallFp<P>> for SmallFp<P> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: &Self) -> Self {
        self -= other;
        self
    }
}

impl<P: SmallFpConfig> Mul<&SmallFp<P>> for SmallFp<P> {
    type Output = Self;

    #[inline]
    fn mul(mut self, other: &Self) -> Self {
        self *= other;
        self
    }
}

impl<P: SmallFpConfig> Div<&SmallFp<P>> for SmallFp<P> {
    type Output = Self;

    /// Returns `self * other.inverse()` if `other.inverse()` is `Some`, and
    /// panics otherwise.
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(mut self, other: &Self) -> Self {
        self *= &other.inverse().unwrap();
        self
    }
}

impl<'b, P: SmallFpConfig> Add<&'b SmallFp<P>> for &SmallFp<P> {
    type Output = SmallFp<P>;

    #[inline]
    fn add(self, other: &'b SmallFp<P>) -> SmallFp<P> {
        let mut result = *self;
        result += other;
        result
    }
}

impl<P: SmallFpConfig> Sub<&SmallFp<P>> for &SmallFp<P> {
    type Output = SmallFp<P>;

    #[inline]
    fn sub(self, other: &SmallFp<P>) -> SmallFp<P> {
        let mut result = *self;
        result -= other;
        result
    }
}

impl<P: SmallFpConfig> Mul<&SmallFp<P>> for &SmallFp<P> {
    type Output = SmallFp<P>;

    #[inline]
    fn mul(self, other: &SmallFp<P>) -> SmallFp<P> {
        let mut result = *self;
        result *= other;
        result
    }
}

impl<P: SmallFpConfig> Div<&SmallFp<P>> for &SmallFp<P> {
    type Output = SmallFp<P>;

    #[inline]
    fn div(self, other: &SmallFp<P>) -> SmallFp<P> {
        let mut result = *self;
        result.div_assign(other);
        result
    }
}

impl<P: SmallFpConfig> AddAssign<&Self> for SmallFp<P> {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        P::add_assign(self, other)
    }
}

impl<P: SmallFpConfig> SubAssign<&Self> for SmallFp<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        P::sub_assign(self, other);
    }
}

impl<P: SmallFpConfig> core::ops::Add<Self> for SmallFp<P> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: Self) -> Self {
        self += &other;
        self
    }
}

impl<'a, P: SmallFpConfig> core::ops::Add<&'a mut Self> for SmallFp<P> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: &'a mut Self) -> Self {
        self += &*other;
        self
    }
}

impl<P: SmallFpConfig> core::ops::Sub<Self> for SmallFp<P> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: Self) -> Self {
        self -= &other;
        self
    }
}

impl<'a, P: SmallFpConfig> core::ops::Sub<&'a mut Self> for SmallFp<P> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: &'a mut Self) -> Self {
        self -= &*other;
        self
    }
}

impl<P: SmallFpConfig> core::iter::Sum<Self> for SmallFp<P> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}

impl<'a, P: SmallFpConfig> core::iter::Sum<&'a Self> for SmallFp<P> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}

impl<P: SmallFpConfig> core::ops::AddAssign<Self> for SmallFp<P> {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self += &other
    }
}

impl<P: SmallFpConfig> core::ops::SubAssign<Self> for SmallFp<P> {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        *self -= &other
    }
}

impl<'a, P: SmallFpConfig> core::ops::AddAssign<&'a mut Self> for SmallFp<P> {
    #[inline(always)]
    fn add_assign(&mut self, other: &'a mut Self) {
        *self += &*other
    }
}

impl<'a, P: SmallFpConfig> core::ops::SubAssign<&'a mut Self> for SmallFp<P> {
    #[inline(always)]
    fn sub_assign(&mut self, other: &'a mut Self) {
        *self -= &*other
    }
}

impl<P: SmallFpConfig> MulAssign<&Self> for SmallFp<P> {
    fn mul_assign(&mut self, other: &Self) {
        P::mul_assign(self, other)
    }
}

/// Computes `self *= other.inverse()` if `other.inverse()` is `Some`, and
/// panics otherwise.
impl<P: SmallFpConfig> DivAssign<&Self> for SmallFp<P> {
    #[inline(always)]
    fn div_assign(&mut self, other: &Self) {
        *self *= &other.inverse().unwrap();
    }
}

impl<P: SmallFpConfig> core::ops::Mul<Self> for SmallFp<P> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, other: Self) -> Self {
        self *= &other;
        self
    }
}

impl<P: SmallFpConfig> core::ops::Div<Self> for SmallFp<P> {
    type Output = Self;

    #[inline(always)]
    fn div(mut self, other: Self) -> Self {
        self.div_assign(&other);
        self
    }
}

impl<'a, P: SmallFpConfig> core::ops::Mul<&'a mut Self> for SmallFp<P> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, other: &'a mut Self) -> Self {
        self *= &*other;
        self
    }
}

impl<'a, P: SmallFpConfig> core::ops::Div<&'a mut Self> for SmallFp<P> {
    type Output = Self;

    #[inline(always)]
    fn div(mut self, other: &'a mut Self) -> Self {
        self.div_assign(&*other);
        self
    }
}

impl<P: SmallFpConfig> core::iter::Product<Self> for SmallFp<P> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), core::ops::Mul::mul)
    }
}

impl<'a, P: SmallFpConfig> core::iter::Product<&'a Self> for SmallFp<P> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}

impl<P: SmallFpConfig> core::ops::MulAssign<Self> for SmallFp<P> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        *self *= &other
    }
}

impl<'a, P: SmallFpConfig> core::ops::DivAssign<&'a mut Self> for SmallFp<P> {
    #[inline(always)]
    fn div_assign(&mut self, other: &'a mut Self) {
        self.div_assign(&*other)
    }
}

impl<'a, P: SmallFpConfig> core::ops::MulAssign<&'a mut Self> for SmallFp<P> {
    #[inline(always)]
    fn mul_assign(&mut self, other: &'a mut Self) {
        *self *= &*other
    }
}

impl<P: SmallFpConfig> core::ops::DivAssign<Self> for SmallFp<P> {
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        self.div_assign(&other)
    }
}

impl<P: SmallFpConfig> zeroize::Zeroize for SmallFp<P> {
    // The phantom data does not contain element-specific data
    // and thus does not need to be zeroized.
    fn zeroize(&mut self) {
        self.value = P::ZERO.value;
    }
}

impl<P: SmallFpConfig> From<num_bigint::BigUint> for SmallFp<P> {
    #[inline]
    fn from(val: num_bigint::BigUint) -> SmallFp<P> {
        SmallFp::from_le_bytes_mod_order(&val.to_bytes_le())
    }
}

impl<P: SmallFpConfig> From<SmallFp<P>> for num_bigint::BigUint {
    #[inline(always)]
    fn from(other: SmallFp<P>) -> Self {
        other.into_bigint().into()
    }
}

impl<P: SmallFpConfig> From<SmallFp<P>> for BigInt<2> {
    fn from(fp: SmallFp<P>) -> Self {
        fp.into_bigint()
    }
}

impl<P: SmallFpConfig> From<BigInt<2>> for SmallFp<P> {
    fn from(int: BigInt<2>) -> Self {
        Self::from_bigint(int).unwrap()
    }
}
