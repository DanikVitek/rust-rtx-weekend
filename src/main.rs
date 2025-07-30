#![feature(
    portable_simd,
    exact_div,
    maybe_uninit_uninit_array_transpose,
    maybe_uninit_slice,
    maybe_uninit_write_slice
)]

use std::{borrow::Cow, ffi::OsStr, io, path::Path};

use rand::Rng;
use thiserror::Error;

mod camera;
mod color;
mod hit;
mod material;
mod math_util;
mod objects;
mod ray;
mod vec2;
mod vec3;

pub use camera::Camera;
pub use color::Color;
pub use hit::Hit;
pub use material::Material;
pub use ray::Ray;
pub use vec3::Vec3;

use crate::{math_util::small_rng, objects::Sphere};

fn main() -> Result<(), Error> {
    let mut rng = small_rng();

    let image_path = std::env::args_os()
        .skip(1)
        .next()
        .map_or(Cow::Borrowed(OsStr::new("./img.ppm")), Cow::Owned);

    if !image_path.to_string_lossy().ends_with(".ppm") {
        return Err(Error::UnsupportedImageFormat);
    }

    let mut world = Vec::<Sphere>::with_capacity(22 * 22 + 4);

    let ground_material: Material = material::Lambertian::new(Color::GRAY).into();
    world.push(Sphere::new(
        Vec3::NEG_Y_AXIS * 1000.0,
        1000.0,
        ground_material,
    ));

    for a in -11..11_i8 {
        for b in -11..11_i8 {
            let center = Vec3::new(
                a as f64 + 0.9 * rng.random::<f64>(),
                0.2,
                b as f64 + 0.9 * rng.random::<f64>(),
            );
            if center.distance(Vec3::new(4.0, 0.2, 0.0)) > 0.9 {
                let mat = match rng.random::<f64>() {
                    ..0.8 => material::Lambertian::new(Color::from(
                        Vec3::random(&mut rng) * Vec3::random(&mut rng),
                    ))
                    .into(),
                    ..0.95 => material::Metal::new(
                        Color::from((Vec3::random(&mut rng) + Vec3::splat(1.0)) / 2.0),
                        rng.random_range(0.0..=0.5),
                    )
                    .into(),
                    _ => material::Dielectric::new(1.5).into(),
                };
                world.push(Sphere::new(center, 0.2, mat));
            }
        }
    }

    let mat1 = material::Dielectric::new(1.5).into();
    world.push(Sphere::new(Vec3::new(0.0, 1.0, 0.0), 1.0, mat1));

    let mat2 = material::Lambertian::new(Color::new(0.4, 0.2, 0.1)).into();
    world.push(Sphere::new(Vec3::new(-4.0, 1.0, 0.0), 1.0, mat2));

    let mat3 = material::Metal::from_albedo(Color::new(0.7, 0.6, 0.5)).into();
    world.push(Sphere::new(Vec3::new(4.0, 1.0, 0.0), 1.0, mat3));

    let camera = Camera::new(camera::Options {
        v_fov: 20.0,
        look_from: Vec3::new(13., 2., 3.),
        look_at: Vec3::ZERO,
        defocus_angle: 0.6,
        focus_dist: 10.,
        ..Default::default()
    });

    camera.render(Path::new(&*image_path), &*world)?;

    Ok(())
}

#[derive(Debug, Error)]
enum Error {
    #[error("Error: Supports only PPM image format")]
    UnsupportedImageFormat,
    #[error(transparent)]
    Io(#[from] io::Error),
}
