use std::{
    io::{self, Write},
    mem::MaybeUninit,
    path::Path,
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::prelude::*;

use crate::{Color, Ray};
use crate::{Vec3, hit::Hittable};
use crate::{math_util::small_rng, vec2::Vec2};

pub struct Camera {
    aspect_ratio: f64,
    img_size: Vec2<u32>,
    samples_per_pixel: u16,
    max_recursion: usize,
    look_from: Vec3,
    defocus_angle: f64,
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    pixel_00_loc: Vec3,
}

pub struct Options {
    /// Ratio of image width over height
    pub aspect_ratio: f64,
    /// Rendered image width in pixel count
    pub img_dim: ImgDim,

    /// Count of random samples for each pixel
    pub samples_per_pixel: u16,
    /// Maximum number of ray bounces into scene
    pub max_recursion: usize,

    /// Vertical view angle (field of view)
    pub v_fov: f64,

    /// Point camera is looking from
    pub look_from: Vec3,
    /// Point camera is looking at
    pub look_at: Vec3,

    /// Camera-relative "up" direction
    pub up: Vec3,

    /// Variation angle of rays through each pixel
    pub defocus_angle: f64,
    /// Distance from camera look-from point to plane of perfect focus
    pub focus_dist: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            aspect_ratio: 16.0 / 9.0,
            img_dim: ImgDim::Width(1920),
            samples_per_pixel: 100,
            max_recursion: 50,
            v_fov: 90.0,
            look_from: Vec3::ZERO,
            look_at: Vec3::NEG_Z_AXIS,
            up: Vec3::Y_AXIS,
            defocus_angle: 0.0,
            focus_dist: 10.0,
        }
    }
}

pub enum ImgDim {
    Width(u32),
    Height(u32),
}

impl Camera {
    pub fn new(options: Options) -> Self {
        let Options {
            aspect_ratio,
            img_dim,
            samples_per_pixel,
            max_recursion,
            v_fov,
            look_from,
            look_at,
            up,
            defocus_angle,
            focus_dist,
        } = options;

        let img_size = match img_dim {
            ImgDim::Width(w) => Vec2::new(w, {
                let h = (w as f64 / aspect_ratio) as u32;
                if h < 1 { 1 } else { h }
            }),
            ImgDim::Height(h) => Vec2::new(
                {
                    let w = (h as f64 * aspect_ratio) as u32;
                    if w < 1 { 1 } else { w }
                },
                h,
            ),
        };

        let viewport_height = {
            let theta = v_fov.to_radians();
            let h = f64::tan(theta / 2.0);
            2.0 * h * focus_dist
        };
        let viewport_width = viewport_height * (img_size.x() as f64 / img_size.y() as f64);

        let w = (look_from - look_at).normalized();
        let u = up.cross(w).normalized();
        let v = w.cross(u);

        let defocus_radius = focus_dist * f64::tan((defocus_angle / 2.0).to_radians());
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;

        let viewport_u = u * viewport_width;
        let viewport_v = v * viewport_height;

        let pixel_delta_u = viewport_u / img_size.x() as f64;
        let pixel_delta_v = viewport_v / img_size.y() as f64;

        let pixel_00_loc = {
            let viewport_upper_left =
                look_from - (w * focus_dist) - (viewport_u / 2.0) - (viewport_v / 2.0);
            (pixel_delta_u + pixel_delta_v) * 0.5 + viewport_upper_left
        };

        Self {
            look_from,
            aspect_ratio,
            img_size,
            defocus_disk_u,
            defocus_disk_v,
            pixel_delta_u,
            pixel_delta_v,
            pixel_00_loc,
            samples_per_pixel,
            max_recursion,
            defocus_angle,
        }
    }

    #[inline]
    pub fn camera_center(&self) -> Vec3 {
        self.look_from
    }

    pub fn render<H>(&self, image_path: &Path, world: &H) -> io::Result<()>
    where
        H: Hittable + Send + Sync + ?Sized,
    {
        let file = std::fs::File::options()
            .create(true)
            .read(true)
            .write(true)
            .append(false)
            .truncate(false)
            .open(image_path)?;

        let mut header_buf = ArrayWriter::<{ 9 + 5 + 5 }>::new();
        write!(
            header_buf,
            "P6\n{w} {h}\n255\n",
            w = self.img_size.x(),
            h = self.img_size.y(),
        )?;
        let header = header_buf.as_slice();

        let size: usize =
            header.len() + self.img_size.x() as usize * self.img_size.y() as usize * 3;
        file.set_len(size as u64)?;
        debug_assert!({
            let metadata = file.metadata()?;
            metadata.len() == size as u64
        });

        let mut mmap = unsafe { memmap::MmapOptions::new().map_mut(&file)? };
        mmap[..header.len()].copy_from_slice(header);

        let (_prefix, image, _suffix) = mmap[header.len()..].as_simd_mut::<3>();
        unsafe {
            core::hint::assert_unchecked(_prefix.is_empty());
            core::hint::assert_unchecked(_suffix.is_empty());
        }

        let grid = self.factorize_parallelism()?;
        let grid_squared = grid * grid;

        #[cfg(debug_assertions)]
        eprintln!("Grid: {w}x{h} WxH", w = grid.x(), h = grid.y());

        let multi_progress = MultiProgress::new();

        let style = ProgressStyle::with_template("{msg} [{pos}/{len}]").unwrap();

        let global_progress = ProgressBar::new((grid_squared.x() * grid_squared.y()) as u64)
            .with_style(style.clone())
            .with_message("Total");
        multi_progress.insert(0, global_progress.clone());

        let img_ptr = image.as_ptr() as usize;
        image
            .par_chunks_mut(self.img_size.area() as usize / grid_squared.area() as usize)
            .map(|chunk| {
                let kernel_i = ((chunk.as_ptr() as usize - img_ptr) / 3) as u32;
                (kernel_i, chunk)
            })
            .for_each(|(ki, img_chunk)| {
                let kh = img_chunk.len() as u32;
                let progress = ProgressBar::new(kh as u64 * self.samples_per_pixel as u64)
                    .with_style(style.clone())
                    .with_message(format!(
                        "{:?}; i: {ki}, h: {kh}",
                        std::thread::current().id()
                    ));
                multi_progress.add(progress.clone());

                (ki..ki + kh).for_each(|i| {
                    let x = i % self.img_size.x();
                    let y = self.img_size.y() - i / self.img_size.x();
                    let pixel = (0..self.samples_per_pixel)
                        .par_bridge()
                        .map(|_| {
                            let mut rng = small_rng();
                            let ray = self.get_ray(&mut rng, x, y);
                            let pixel = self.ray_color(&mut rng, ray, world, 0);
                            progress.inc(1);
                            pixel
                        })
                        .sum::<Color>()
                        / self.samples_per_pixel as f64;

                    img_chunk[(i - ki) as usize] = pixel.to_gamma_bytes();
                });

                global_progress.inc(1);
            });

        Ok(())
    }

    fn get_ray<R: Rng>(&self, rng: &mut R, x: u32, y: u32) -> Ray {
        let (offset_x, offset_y) = rng.random::<(f64, f64)>();
        let pixel_sample = self.pixel_delta_u.mul_scalar_add(
            x as f64 + offset_x,
            self.pixel_delta_v
                .mul_scalar_add(y as f64 + offset_y, self.pixel_00_loc),
        );

        let ray_origin = if self.defocus_angle <= 0.0 {
            self.camera_center()
        } else {
            self.defocus_disk_sample(rng)
        };
        let ray_direction = pixel_sample - ray_origin;

        Ray::new(ray_origin, ray_direction)
    }

    fn defocus_disk_sample<R: Rng>(&self, rng: &mut R) -> Vec3 {
        let p = Vec2::random_unit(rng);
        self.defocus_disk_u.mul_scalar_add(
            p.x(),
            self.defocus_disk_v
                .mul_scalar_add(p.y(), self.camera_center()),
        )
    }

    fn ray_color<R, H>(&self, rng: &mut R, ray: Ray, world: &H, depth: usize) -> Color
    where
        H: Hittable + ?Sized,
        R: Rng,
    {
        if depth > self.max_recursion {
            return Color::BLACK;
        }

        match world.hit(ray, ..) {
            Some(hit) => match hit.material().scatter(rng, ray, &hit) {
                Some(scatter) => {
                    scatter.attenuation
                        * self.ray_color(rng, scatter.scattered_ray, world, depth + 1)
                }
                None => Color::BLACK,
            },
            None => {
                let unit_direction = ray.dir().normalized();
                let a = 0.5 * (unit_direction.y() + 1.0);
                let wat = Color::new(0.5, 0.7, 1.0);
                wat.mul_scalar_add(a, Color::splat(1.0 - a))
            }
        }
    }

    fn factorize_parallelism(&self) -> Result<Vec2<GridSide>, io::Error> {
        let parallelism = std::thread::available_parallelism()?.get();

        let dim_a = parallelism.isqrt() as GridSide;
        let dim_b = parallelism.exact_div(dim_a as usize) as GridSide;

        Ok(Vec2::new(
            if self.aspect_ratio > 1.0 {
                dim_a.max(dim_b)
            } else {
                dim_a.min(dim_b)
            },
            if self.aspect_ratio > 1.0 {
                dim_a.min(dim_b)
            } else {
                dim_a.max(dim_b)
            },
        ))
    }
}

#[cfg(target_pointer_width = "64")]
type GridSide = u32;

#[cfg(target_pointer_width = "32")]
type GridSide = u16;

struct ArrayWriter<const N: usize> {
    buf: [MaybeUninit<u8>; N],
    written: usize,
}

impl<const N: usize> Write for ArrayWriter<N> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.written == N {
            Ok(0)
        } else {
            let remaining = N - self.written;
            let to_write = std::cmp::min(buf.len(), remaining);
            self.buf[self.written..][..to_write].write_copy_of_slice(&buf[..to_write]);
            self.written += to_write;
            Ok(to_write)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.written = 0;
        Ok(())
    }
}

impl<const N: usize> ArrayWriter<N> {
    fn new() -> Self {
        Self {
            buf: MaybeUninit::<[u8; _]>::uninit().transpose(),
            written: 0,
        }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { self.buf[..self.written].assume_init_ref() }
    }
}

// #[derive(Debug, thiserror::Error)]
// pub enum RenderError {
//     #[error(transparent)]
//     Io(#[from] io::Error),
// }
