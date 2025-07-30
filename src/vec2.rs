use core::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    simd::Simd,
};
use std::{
    ops::{Neg, Rem, RemAssign},
    simd::{SimdElement, StdFloat, cmp::SimdPartialOrd, num::SimdFloat},
};

use rand::Rng;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec2<T: SimdElement = f64>(pub Simd<T, 2>);

impl Vec2 {
    pub const ZERO: Self = Self::splat(0.);
    pub const ONES: Self = Self::splat(1.);
    pub const X_AXIS: Self = Self::new(1., 0.);
    pub const Y_AXIS: Self = Self::new(0., 1.);
    pub const NEG_X_AXIS: Self = Self::new(-1., 0.);
    pub const NEG_Y_AXIS: Self = Self::new(0., -1.);

    pub fn random_unit(rng: &mut impl Rng) -> Self {
        loop {
            let v = Self::random(rng);
            let l2 = v.length_squared();
            if const { 0f64.next_up() } < l2 && l2 <= 1.0 {
                return v / l2.sqrt();
            }
        }
    }

    pub fn random_unit_in_hemisphere(rng: &mut impl Rng, normal: Self) -> Self {
        let point = Self::random_unit(rng);
        if point.dot(normal) < 0.0 {
            -point
        } else {
            point
        }
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            return Self::ZERO;
        }
        self / len
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    pub fn is_near_zero(self) -> bool {
        self.0.abs().simd_lt(Simd::splat(1e-8)).all()
    }

    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    pub fn mul_scalar_add(self, scalar: f64, other: Self) -> Self {
        Self(self.0.mul_add(Simd::splat(scalar), other.0))
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    fn distance(self, other: Self) -> f64 {
        self.distance_squared(other).sqrt()
    }

    pub fn dot(self, other: Self) -> f64 {
        (self.0 * other.0).reduce_sum()
    }

    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn distance_squared(self, other: Self) -> f64 {
        (self - other).length_squared()
    }
}

impl<T: SimdElement> Vec2<T> {
    pub const fn new(x: T, y: T) -> Self {
        Self(Simd::from_array([x, y]))
    }

    pub const fn splat(value: T) -> Self {
        Self(Simd::splat(value))
    }

    pub const fn x(self) -> T {
        self.0.as_array()[0]
    }

    pub const fn y(self) -> T {
        self.0.as_array()[1]
    }
}

impl<T: SimdElement> Vec2<T>
where
    T: Mul<Output = T>,
{
    pub fn area(self) -> T {
        self.x() * self.y()
    }
}

impl<T: SimdElement> Neg for Vec2<T>
where
    Simd<T, 2>: Neg<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<T: SimdElement> Add for Vec2<T>
where
    Simd<T, 2>: Add<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl<T: SimdElement> AddAssign for Vec2<T>
where
    Simd<T, 2>: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl<T: SimdElement> Sub for Vec2<T>
where
    Simd<T, 2>: Sub<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl<T: SimdElement> SubAssign for Vec2<T>
where
    Simd<T, 2>: SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl<T: SimdElement> Mul for Vec2<T>
where
    Simd<T, 2>: Mul<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl<T: SimdElement> MulAssign for Vec2<T>
where
    Simd<T, 2>: MulAssign,
{
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl<T: SimdElement> Mul<T> for Vec2<T>
where
    Simd<T, 2>: Mul<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self(self.0 * Simd::splat(scalar))
    }
}

impl<T: SimdElement> MulAssign<T> for Vec2<T>
where
    Simd<T, 2>: Mul<Output = Simd<T, 2>>,
{
    fn mul_assign(&mut self, scalar: T) {
        self.0 *= Simd::splat(scalar);
    }
}

macro_rules! impl_mul_for_scalar {
    ($($scalar:ty)*) => {
        $(
            impl Mul<Vec2<$scalar>> for $scalar {
                type Output = Vec2<$scalar>;

                fn mul(self, vec: Vec2<$scalar>) -> Vec2<$scalar> {
                    Vec2(Simd::splat(self) * vec.0)
                }
            }
        )*
    };
}

impl_mul_for_scalar!(u8 i8 u16 i16 u32 i32 u64 i64 usize isize f32 f64);

impl<T: SimdElement> Div<T> for Vec2<T>
where
    Simd<T, 2>: Div<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self(self.0 / Simd::splat(scalar))
    }
}

impl<T: SimdElement> DivAssign<T> for Vec2<T>
where
    Simd<T, 2>: Div<Output = Simd<T, 2>>,
{
    fn div_assign(&mut self, scalar: T) {
        self.0 /= Simd::splat(scalar);
    }
}

impl<T: SimdElement> Div for Vec2<T>
where
    Simd<T, 2>: Div<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl<T: SimdElement> DivAssign for Vec2<T>
where
    Simd<T, 2>: Div<Output = Simd<T, 2>>,
{
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<T: SimdElement> Rem for Vec2<T>
where
    Simd<T, 2>: Rem<Output = Simd<T, 2>>,
{
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Self(self.0 % other.0)
    }
}

impl<T: SimdElement> RemAssign for Vec2<T>
where
    Simd<T, 2>: Rem<Output = Simd<T, 2>>,
{
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}
