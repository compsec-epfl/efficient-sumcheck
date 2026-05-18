//! Exported convenience macros.

/// Define a thin newtype wrapper around a field type and implement
/// [`SumcheckField`](crate::field::SumcheckField) for it.
///
/// This is useful when integrating a field type from another ecosystem whose
/// trait impls cannot be implemented directly because of Rust's orphan rules.
///
/// ```ignore
/// effsc::sumcheck_field_newtype! {
///     pub struct MyField(OtherCrateField);
///     const ZERO = OtherCrateField::ZERO;
///     const ONE = OtherCrateField::ONE;
///     fn from_u64(val) { OtherCrateField::from_canonical_u64(val) }
///     fn inverse(self) { self.0.try_inverse() }
/// }
/// ```
#[macro_export]
macro_rules! sumcheck_field_newtype {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident($inner:ty);
        const ZERO = $zero:expr;
        const ONE = $one:expr;
        fn from_u64($from_arg:ident) $from_u64:block
        fn inverse($self_arg:ident) $inverse:block
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Copy, Clone, Debug, PartialEq)]
        $vis struct $name($inner);

        impl $name {
            #[inline]
            pub fn new(val: u64) -> Self {
                <Self as $crate::field::SumcheckField>::from_u64(val)
            }

            #[inline]
            pub const fn from_inner(inner: $inner) -> Self {
                Self(inner)
            }

            #[inline]
            pub fn into_inner(self) -> $inner {
                self.0
            }
        }

        impl core::fmt::Display for $name {
            #[inline]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{:?}", self.0)
            }
        }

        impl core::ops::Add for $name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl core::ops::Sub for $name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl core::ops::Mul for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }

        impl core::ops::Neg for $name {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl core::ops::AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl core::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl core::ops::MulAssign for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0;
            }
        }

        impl core::iter::Sum for $name {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<Self as $crate::field::SumcheckField>::ZERO, |acc, x| acc + x)
            }
        }

        impl $crate::field::SumcheckField for $name {
            const ZERO: Self = Self($zero);
            const ONE: Self = Self($one);

            #[inline]
            fn from_u64($from_arg: u64) -> Self {
                Self($from_u64)
            }

            #[inline]
            fn inverse(&$self_arg) -> Option<Self> {
                $inverse.map(Self)
            }
        }
    };
}
