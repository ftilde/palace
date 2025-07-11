use crate::{dim::*, mat::Matrix, vec::Vector};

pub struct AABB<D: DynDimension, T: Copy> {
    min: Vector<D, T>,
    max: Vector<D, T>,
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

impl<D: DynDimension, T: Copy + PartialOrd> AABB<D, T> {
    pub fn new(p1: &Vector<D, T>, p2: &Vector<D, T>) -> Self {
        Self {
            min: p1.zip(p2, partial_ord_min),
            max: p1.zip(p2, partial_ord_max),
        }
    }

    pub fn from_points(mut points: impl Iterator<Item = Vector<D, T>>) -> Self {
        let first = points.next().unwrap();
        let mut s = Self {
            min: first.clone(),
            max: first,
        };
        for p in points {
            s.add_point(&p);
        }
        s
    }

    pub fn add_point(&mut self, p: &Vector<D, T>) {
        assert_eq!(self.dim(), p.dim());
        self.min = self.min.zip(p, partial_ord_min);
        self.max = self.max.zip(p, partial_ord_max);
    }

    pub fn lower(&self) -> &Vector<D, T> {
        &self.min
    }

    pub fn upper(&self) -> &Vector<D, T> {
        &self.max
    }

    pub fn dim(&self) -> D {
        self.min.dim()
    }

    //pub fn contains(&self, p: Vector<D, T>) -> bool {
    //    let bigger_than_min = self.min.zip(p, |v1, v2| v1.le(&v2));
    //    let smaller_than_max = p.zip(self.max, |v1, v2| v1.lt(&v2));
    //    bigger_than_min
    //        .0
    //        .iter()
    //        .chain(smaller_than_max.0.iter())
    //        .all(|v| *v)
    //}
}
impl<D: LargerDim> AABB<D, f32> {
    #[must_use]
    pub fn transform(&self, t: &Matrix<D::Larger, f32>) -> Self {
        let points = (0..(1 << self.dim().n())).into_iter().map(|b| {
            let p = Vector::<D, f32>::try_from_fn_and_len(self.min.len(), |i| {
                if (b & (1 << i)) != 0 {
                    self.min[i]
                } else {
                    self.max[i]
                }
            })
            .unwrap();
            t.clone().transform(&p)
        });
        Self::from_points(points)
    }
}
