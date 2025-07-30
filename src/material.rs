use rand::Rng;

use crate::{Color, Hit, Ray, Vec3};

#[derive(Debug, Clone, Copy)]
pub enum Material {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
}

#[derive(Debug, Clone, Copy)]
pub struct Lambertian {
    albedo: Color,
}

#[derive(Debug, Clone, Copy)]
pub struct Metal {
    albedo: Color,
    fuzz: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Dielectric {
    ref_idx: f64,
}

impl Lambertian {
    pub const fn new(albedo: Color) -> Self {
        Self { albedo }
    }

    pub const fn albedo(self) -> Color {
        self.albedo
    }
}

impl Metal {
    pub const fn new(albedo: Color, fuzz: f64) -> Self {
        Self { albedo, fuzz }
    }

    #[inline]
    pub const fn from_albedo(albedo: Color) -> Self {
        Self::new(albedo, 0.0)
    }

    pub const fn albedo(self) -> Color {
        self.albedo
    }

    pub const fn fuzz(self) -> f64 {
        self.fuzz
    }
}

impl Dielectric {
    pub const fn new(ref_idx: f64) -> Self {
        Self { ref_idx }
    }

    pub const fn ref_idx(self) -> f64 {
        self.ref_idx
    }
}

impl From<Lambertian> for Material {
    #[inline]
    fn from(lambertian: Lambertian) -> Self {
        Self::Lambertian(lambertian)
    }
}

impl From<Metal> for Material {
    #[inline]
    fn from(metal: Metal) -> Self {
        Self::Metal(metal)
    }
}

impl From<Dielectric> for Material {
    #[inline]
    fn from(dielectric: Dielectric) -> Self {
        Self::Dielectric(dielectric)
    }
}

pub struct Scatter {
    pub scattered_ray: Ray,
    pub attenuation: Color,
}

impl Material {
    pub fn scatter(self, rng: &mut impl Rng, ray_in: Ray, hit: &Hit) -> Option<Scatter> {
        match self {
            Self::Lambertian(m) => Some(m.scatter(rng, hit)),
            Self::Metal(m) => m.scatter(rng, ray_in, hit),
            Self::Dielectric(m) => Some(m.scatter(rng, ray_in, hit)),
        }
    }
}

impl Lambertian {
    pub fn scatter(self, rng: &mut impl Rng, hit: &Hit) -> Scatter {
        let mut scatter_dir = hit.norm() + Vec3::random_unit(rng);

        if scatter_dir.is_near_zero() {
            scatter_dir = hit.norm();
        }

        Scatter {
            scattered_ray: Ray::new(hit.pos(), scatter_dir),
            attenuation: self.albedo,
        }
    }
}

impl Metal {
    pub fn scatter(self, rng: &mut impl Rng, ray_in: Ray, hit: &Hit) -> Option<Scatter> {
        let mut reflected = ray_in.dir().reflect(hit.norm());
        reflected = Vec3::random_unit(rng).mul_scalar_add(self.fuzz, reflected.normalized());
        (reflected.dot(hit.norm()) > 0.0).then(|| Scatter {
            scattered_ray: Ray::new(hit.pos(), reflected),
            attenuation: self.albedo,
        })
    }
}

impl Dielectric {
    pub fn scatter(self, rng: &mut impl Rng, ray_in: Ray, hit: &Hit) -> Scatter {
        let ri = if hit.front_face() {
            self.ref_idx.recip()
        } else {
            self.ref_idx
        };

        let dir = ray_in.dir().normalized();
        let refracted = dir.refract(rng, hit.norm(), ri);

        Scatter {
            scattered_ray: Ray::new(hit.pos(), refracted),
            attenuation: Color::WHITE,
        }
    }
}
