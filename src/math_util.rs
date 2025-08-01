use core::cell::UnsafeCell;
use std::rc::Rc;

use rand::{RngCore, SeedableRng, rngs::SmallRng};

pub const fn approx_eq_abs(x: f64, y: f64, tolerance: f64) -> bool {
    debug_assert!(tolerance >= 0.0);

    if x == y {
        return true;
    }

    if x.is_nan() || y.is_nan() {
        return false;
    }

    (x - y).abs() <= tolerance
}

thread_local! {
    static RNG: Rc<UnsafeCell<SmallRng>> = Rc::new(UnsafeCell::new(SmallRng::seed_from_u64(0)))
}

pub struct SmallThreadRng {
    rng: Rc<UnsafeCell<SmallRng>>,
}

pub fn small_rng() -> SmallThreadRng {
    let rng = RNG.with(|rng| rng.clone());
    SmallThreadRng { rng }
}

impl RngCore for SmallThreadRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        unsafe { &mut *self.rng.get() }.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        unsafe { &mut *self.rng.get() }.next_u64()
    }

    #[inline(always)]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        unsafe { &mut *self.rng.get() }.fill_bytes(dst)
    }
}
