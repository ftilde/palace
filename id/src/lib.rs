use sha1_smol::Sha1;
use std::hash::Hash;

pub use id_derive::Identify;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Id(u128);

#[repr(C, align(16))]
struct IdBytes([u8; 16]);

fn digest_to_u128(v: [u8; 20]) -> u128 {
    let bytes = IdBytes(std::array::from_fn(|i| v[i]));
    *bytemuck::from_bytes(&bytes.0)
}

impl Id {
    pub fn raw(&self) -> u128 {
        self.0
    }
    pub fn from_data(data: &[u8]) -> Self {
        let sha = Sha1::from(data);
        Self(digest_to_u128(sha.digest().bytes()))
    }
    pub fn hash<T: std::hash::Hash + ?Sized>(data: &T) -> Self {
        let mut hasher = xxhash_rust::xxh3::Xxh3Builder::new().with_seed(0).build();
        data.hash(&mut hasher);
        let digest = hasher.digest128();
        let hash = bytemuck::bytes_of(&digest);
        Id::from_data(hash)
    }
    pub fn combine(ids: &[Id]) -> Self {
        Self::combine_it(ids.into_iter().cloned())
    }
    pub fn combine_it(ids: impl Iterator<Item = Id>) -> Self {
        let mut sha = Sha1::new();
        for id in ids {
            sha.update(bytemuck::bytes_of(&id.0));
        }
        Self(digest_to_u128(sha.digest().bytes()))
    }
}

impl From<&[u8]> for Id {
    fn from(value: &[u8]) -> Self {
        Self::from_data(value)
    }
}

pub trait Identify {
    fn id(&self) -> Id;
}

impl<I: Identify> Identify for &I {
    fn id(&self) -> Id {
        I::id(self)
    }
}

impl<I1: Identify, I2: Identify> Identify for (I1, I2) {
    fn id(&self) -> Id {
        Id::combine(&[self.0.id(), self.1.id()])
    }
}
impl<I1: Identify, I2: Identify, I3: Identify> Identify for (I1, I2, I3) {
    fn id(&self) -> Id {
        Id::combine(&[self.0.id(), self.1.id(), self.2.id()])
    }
}
impl<I1: Identify, I2: Identify, I3: Identify, I4: Identify> Identify for (I1, I2, I3, I4) {
    fn id(&self) -> Id {
        Id::combine(&[self.0.id(), self.1.id(), self.2.id(), self.3.id()])
    }
}
impl<I1: Identify, I2: Identify, I3: Identify, I4: Identify, I5: Identify> Identify
    for (I1, I2, I3, I4, I5)
{
    fn id(&self) -> Id {
        Id::combine(&[
            self.0.id(),
            self.1.id(),
            self.2.id(),
            self.3.id(),
            self.4.id(),
        ])
    }
}
impl<I1: Identify, I2: Identify, I3: Identify, I4: Identify, I5: Identify, I6: Identify> Identify
    for (I1, I2, I3, I4, I5, I6)
{
    fn id(&self) -> Id {
        Id::combine(&[
            self.0.id(),
            self.1.id(),
            self.2.id(),
            self.3.id(),
            self.4.id(),
            self.5.id(),
        ])
    }
}

impl<
        I1: Identify,
        I2: Identify,
        I3: Identify,
        I4: Identify,
        I5: Identify,
        I6: Identify,
        I7: Identify,
    > Identify for (I1, I2, I3, I4, I5, I6, I7)
{
    fn id(&self) -> Id {
        Id::combine(&[
            self.0.id(),
            self.1.id(),
            self.2.id(),
            self.3.id(),
            self.4.id(),
            self.5.id(),
            self.6.id(),
        ])
    }
}

impl<
        I1: Identify,
        I2: Identify,
        I3: Identify,
        I4: Identify,
        I5: Identify,
        I6: Identify,
        I7: Identify,
        I8: Identify,
    > Identify for (I1, I2, I3, I4, I5, I6, I7, I8)
{
    fn id(&self) -> Id {
        Id::combine(&[
            self.0.id(),
            self.1.id(),
            self.2.id(),
            self.3.id(),
            self.4.id(),
            self.5.id(),
            self.6.id(),
            self.7.id(),
        ])
    }
}

pub struct IdentifyHash<T>(pub T);

impl<T: Hash> Identify for IdentifyHash<T> {
    fn id(&self) -> Id {
        Id::hash(&self.0)
    }
}

impl<T> std::ops::Deref for IdentifyHash<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Identify for Id {
    fn id(&self) -> Id {
        *self
    }
}

impl<E: Identify> Identify for [E] {
    fn id(&self) -> Id {
        Id::combine_it(self.iter().map(|v| v.id()))
    }
}

impl<E: Identify> Identify for Vec<E> {
    fn id(&self) -> Id {
        self.as_slice().id()
    }
}

impl<E: Identify> Identify for Box<E> {
    fn id(&self) -> Id {
        self.as_ref().id()
    }
}

impl<E: Identify> Identify for Option<E> {
    fn id(&self) -> Id {
        match self {
            Some(n) => Id::combine_it([Id::from_data(&[1]), n.id()].into_iter()),
            None => Id::from_data(&[0]),
        }
    }
}

impl Identify for str {
    fn id(&self) -> Id {
        Id::from_data(self.as_bytes())
    }
}

impl Identify for &str {
    fn id(&self) -> Id {
        Id::from_data(self.as_bytes())
    }
}

impl Identify for String {
    fn id(&self) -> Id {
        self.as_str().id()
    }
}

macro_rules! impl_pod {
    ($ty:ty) => {
        impl Identify for $ty {
            fn id(&self) -> Id {
                Id::from_data(bytemuck::bytes_of(self))
            }
        }
    };
}

impl_pod!(u8);
impl_pod!(u16);
impl_pod!(u32);
impl_pod!(u64);
impl_pod!(i8);
impl_pod!(i16);
impl_pod!(i32);
impl_pod!(i64);
impl_pod!(usize);
impl_pod!(f32);
impl_pod!(f64);
impl_pod!(());

pub fn func_id<F: 'static>() -> Id {
    // TODO: One problem with this during development: The id of a closure may not change between
    // compilations, which may result in confusion when old values are reported when loaded from a
    // persistent cache even though the closure code was changed in the meantime.
    let id = std::any::TypeId::of::<F>();
    Id::hash(&id)
}
