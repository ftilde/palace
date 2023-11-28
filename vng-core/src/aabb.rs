use crate::{mat::Matrix, vec::Vector};

pub struct AABB<const N: usize, T> {
    min: Vector<N, T>,
    max: Vector<N, T>,
}

fn partial_ord_min<T: PartialOrd>(v1: T, v2: T) -> T {
    if v1.lt(&v2) {
        v1
    } else {
        v2
    }
}
fn partial_ord_max<T: PartialOrd>(v1: T, v2: T) -> T {
    if v1.lt(&v2) {
        v2
    } else {
        v1
    }
}

impl<const N: usize, T: Copy + PartialOrd> AABB<N, T> {
    pub fn new(p1: Vector<N, T>, p2: Vector<N, T>) -> Self {
        Self {
            min: p1.zip(p2, partial_ord_min),
            max: p1.zip(p2, partial_ord_max),
        }
    }

    pub fn from_points(mut points: impl Iterator<Item = Vector<N, T>>) -> Self {
        let first = points.next().unwrap();
        let mut s = Self {
            min: first,
            max: first,
        };
        for p in points {
            s.add_point(p);
        }
        s
    }

    pub fn add_point(&mut self, p: Vector<N, T>) {
        self.min = self.min.zip(p, partial_ord_min);
        self.max = self.max.zip(p, partial_ord_max);
    }

    pub fn lower(&self) -> Vector<N, T> {
        self.min
    }

    pub fn upper(&self) -> Vector<N, T> {
        self.max
    }

    //pub fn contains(&self, p: Vector<N, T>) -> bool {
    //    let bigger_than_min = self.min.zip(p, |v1, v2| v1.le(&v2));
    //    let smaller_than_max = p.zip(self.max, |v1, v2| v1.lt(&v2));
    //    bigger_than_min
    //        .0
    //        .iter()
    //        .chain(smaller_than_max.0.iter())
    //        .all(|v| *v)
    //}
}
impl AABB<3, f32> {
    #[must_use]
    pub fn transform(&self, t: &Matrix<4, f32>) -> Self {
        let points = (0..8).into_iter().map(|b| {
            let p = Vector::<3, f32>::from_fn(|i| {
                if (b & (1 << i)) != 0 {
                    self.min[i]
                } else {
                    self.max[i]
                }
            });
            (*t * p.to_homogeneous_coord()).drop_dim(0)
        });
        Self::from_points(points)
    }
}
