use std::{
    f64,
    ops::{Bound, RangeBounds},
};

use crate::{Material, Ray, Vec3, math_util::approx_eq_abs};

#[derive(Debug, Clone)]
pub struct Hit {
    pos: Vec3,
    norm: Vec3,
    t: f64,
    front_face: bool,
    material: Material,
}

impl Hit {
    pub fn new(t: f64, pos: Vec3, ray: Ray, outward_normal: Vec3, material: Material) -> Self {
        debug_assert!(approx_eq_abs(outward_normal.length_squared(), 1.0, 1e-8));

        let front_face = ray.dir().dot(outward_normal) < 0.0;
        let norm = if front_face {
            outward_normal
        } else {
            -outward_normal
        };

        Self {
            pos,
            norm,
            t,
            front_face,
            material,
        }
    }

    pub fn pos(&self) -> Vec3 {
        self.pos
    }

    pub fn norm(&self) -> Vec3 {
        self.norm
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn front_face(&self) -> bool {
        self.front_face
    }

    pub fn material(&self) -> Material {
        self.material
    }
}

pub trait Hittable {
    fn hit(&self, ray: Ray, ray_t: impl RangeBounds<f64>) -> Option<Hit>;
}

impl<T: Hittable> Hittable for &T {
    fn hit(&self, ray: Ray, ray_t: impl RangeBounds<f64>) -> Option<Hit> {
        (*self).hit(ray, ray_t)
    }
}

impl<T: Hittable> Hittable for &mut T {
    fn hit(&self, ray: Ray, ray_t: impl RangeBounds<f64>) -> Option<Hit> {
        (&**self).hit(ray, ray_t)
    }
}

impl<T> Hittable for [T]
where
    for<'a> &'a T: Hittable,
{
    fn hit(&self, ray: Ray, ray_t: impl RangeBounds<f64>) -> Option<Hit> {
        let mut hit = None;
        let ray_t_min = match ray_t.start_bound() {
            Bound::Included(x) | Bound::Excluded(x) => x.max(0.001),
            Bound::Unbounded => 0.001,
        };
        let mut closest_so_far = match ray_t.end_bound() {
            Bound::Included(x) | Bound::Excluded(x) => *x,
            Bound::Unbounded => f64::INFINITY,
        };

        for object in self {
            if let Some(h) = object.hit(ray, ray_t_min..=closest_so_far) {
                closest_so_far = h.t;
                hit = Some(h);
            }
        }

        hit
    }
}
