use crate::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn orig(self) -> Vec3 {
        self.origin
    }

    pub fn dir(self) -> Vec3 {
        self.direction
    }

    pub fn at(self, t: f64) -> Vec3 {
        self.origin + t * self.direction
    }
}
