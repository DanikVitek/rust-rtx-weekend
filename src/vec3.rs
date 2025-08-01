use core::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    simd::Simd,
};
use std::{
    ops::Neg,
    simd::{StdFloat, cmp::SimdPartialOrd, num::SimdFloat},
};

use rand::Rng;

use crate::math_util::approx_eq_abs;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3(pub Simd<f64, 3>);

impl Vec3 {
    pub const ZERO: Self = Self::splat(0.);
    pub const ONES: Self = Self::splat(1.);
    pub const X_AXIS: Self = Self::new(1., 0., 0.);
    pub const Y_AXIS: Self = Self::new(0., 1., 0.);
    pub const Z_AXIS: Self = Self::new(0., 0., 1.);
    pub const NEG_X_AXIS: Self = Self::new(-1., 0., 0.);
    pub const NEG_Y_AXIS: Self = Self::new(0., -1., 0.);
    pub const NEG_Z_AXIS: Self = Self::new(0., 0., -1.);

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self(Simd::from_array([x, y, z]))
    }

    pub const fn splat(value: f64) -> Self {
        Self(Simd::splat(value))
    }

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
        Self::new(
            rng.random_range(-1.0..=1.0),
            rng.random_range(-1.0..=1.0),
            rng.random_range(-1.0..=1.0),
        )
    }

    pub const fn x(self) -> f64 {
        self.0.as_array()[0]
    }

    pub const fn y(self) -> f64 {
        self.0.as_array()[1]
    }

    pub const fn z(self) -> f64 {
        self.0.as_array()[2]
    }

    pub fn dot(self, other: Self) -> f64 {
        (self.0 * other.0).reduce_sum()
    }

    pub const fn cross(self, other: Self) -> Self {
        Self::new(
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        )
    }

    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }

    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    pub fn mul_scalar_add(self, scalar: f64, other: Self) -> Self {
        Self(self.0.mul_add(Simd::splat(scalar), other.0))
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

    pub fn reflect(self, norm: Self) -> Self {
        self - (2.0 * self.dot(norm) * norm)
    }

    pub fn refract(self, rng: &mut impl Rng, norm: Self, refraction_idx: f64) -> Self {
        debug_assert!(approx_eq_abs(self.length_squared(), 1.0, 1e-8));

        let cos_theta = (-self).dot(norm).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract: bool = refraction_idx * sin_theta > 1.0;
        if cannot_refract || reflectance(cos_theta, refraction_idx) > rng.random::<f64>() {
            return self.reflect(norm);
        }

        let r_out_perp = norm.mul_scalar_add(cos_theta, self) * refraction_idx;
        let r_out_parallel = norm * -(1.0 - r_out_perp.length_squared()).abs().sqrt();
        return r_out_perp.add(r_out_parallel);
    }

    pub fn is_near_zero(self) -> bool {
        self.0.abs().simd_lt(Simd::splat(1e-8)).all()
    }
}

fn reflectance(cos_theta: f64, refraction_idx: f64) -> f64 {
    let r0 = (1.0 - refraction_idx) / (1.0 + refraction_idx);
    let r02 = r0 * r0;
    r02 + (1.0 - r02) * (1.0 - cos_theta).powi(5)
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl Mul for Vec3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self(self.0 * Simd::splat(scalar))
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, scalar: f64) {
        self.0 *= Simd::splat(scalar);
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        vec * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Self(self.0 / Simd::splat(scalar))
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, scalar: f64) {
        self.0 /= Simd::splat(scalar);
    }
}

impl Div for Vec3 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl DivAssign for Vec3 {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}
