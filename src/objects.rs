use crate::{Hit, Material, Vec3, hit::Hittable};

pub struct Sphere {
    center: Vec3,
    radius: f64,
    material: Material,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, material: Material) -> Self {
        debug_assert!(radius >= 0.0);
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: crate::Ray, ray_t: impl std::ops::RangeBounds<f64>) -> Option<Hit> {
        let center = self.center;
        let radius = self.radius;
        let r_dir = ray.dir();

        let oc = center - ray.orig();
        let a = r_dir.length_squared();
        let h = r_dir.dot(oc);
        let c = oc.length_squared() - radius * radius;
        let discriminant = h * h - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        let mut eq_root = (h - sqrtd) / a;
        if !ray_t.contains(&eq_root) {
            eq_root = (h + sqrtd) / a;
            if !ray_t.contains(&eq_root) {
                return None;
            }
        }

        let pos = ray.at(eq_root);
        let outward_normal = (pos - center) / radius;

        Some(Hit::new(eq_root, pos, ray, outward_normal, self.material))
    }
}
