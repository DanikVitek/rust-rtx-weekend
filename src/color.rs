use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::{
    io::{self, Write},
    iter::Sum,
    simd::{Simd, StdFloat, num::SimdFloat},
};

use crate::Vec3;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Color(pub Vec3);

impl From<Vec3> for Color {
    fn from(vec: Vec3) -> Self {
        Self(vec)
    }
}

impl Color {
    pub const BLACK: Self = Self(Vec3::ZERO);
    pub const GRAY: Self = Self(Vec3::splat(0.5));
    pub const WHITE: Self = Self(Vec3::ONES);
    pub const RED: Self = Self(Vec3::X_AXIS);
    pub const GREEN: Self = Self(Vec3::Y_AXIS);
    pub const BLUE: Self = Self(Vec3::Z_AXIS);

    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self(Vec3::new(r, g, b))
    }

    pub const fn splat(value: f64) -> Self {
        Self(Vec3::splat(value))
    }

    pub const fn r(self) -> f64 {
        self.0.x()
    }

    pub const fn g(self) -> f64 {
        self.0.y()
    }

    pub const fn b(self) -> f64 {
        self.0.z()
    }

    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    #[inline]
    pub fn mul_scalar_add(self, scalar: f64, other: Self) -> Self {
        Self(self.0.mul_scalar_add(scalar, other.0))
    }

    pub fn to_gamma_bytes(self) -> Simd<u8, 3> {
        (self
            .0
            .0
            .simd_clamp(const { Simd::splat(0.) }, const { Simd::splat(1.) })
            .sqrt()
            * const { Simd::splat(255.) })
        .cast::<u8>()
    }
}

// fn linear_to_gamma(linear: Simd<f64, 3>) -> Simd<f64, 3> {
//     linear.sqrt()
// }

impl Add for Color {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Color {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul for Color {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl Mul<f64> for Color {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self {
        Self(self.0 * scalar)
    }
}

impl Div for Color {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl Div<f64> for Color {
    type Output = Self;

    #[inline]
    fn div(self, scalar: f64) -> Self {
        Self(self.0 / scalar)
    }
}

impl AddAssign for Color {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl SubAssign for Color {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for Color {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl MulAssign<f64> for Color {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.0 *= scalar;
    }
}

impl DivAssign for Color {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl DivAssign<f64> for Color {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        self.0 /= scalar;
    }
}

impl Sum for Color {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Color(Vec3(iter.map(|c| c.0.0).sum()))
    }
}
