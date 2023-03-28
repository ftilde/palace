use sha1_smol::Sha1;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Id(u128);

#[repr(C, align(16))]
struct IdBytes([u8; 16]);

fn digest_to_u128(v: [u8; 20]) -> u128 {
    let bytes = IdBytes(std::array::from_fn(|i| v[i]));
    *bytemuck::from_bytes(&bytes.0)
}

impl Id {
    pub fn from_data(data: &[u8]) -> Self {
        let sha = Sha1::from(data);
        Self(digest_to_u128(sha.digest().bytes()))
    }
    pub fn hash(data: &impl std::hash::Hash) -> Self {
        let mut hasher = xxhash_rust::xxh3::Xxh3Builder::new().with_seed(0).build();
        data.hash(&mut hasher);
        let digest = hasher.digest128();
        let hash = bytemuck::bytes_of(&digest);
        crate::id::Id::from_data(hash)
    }
    pub fn combine(ids: &[Id]) -> Self {
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

pub fn func_id<F: 'static>() -> Id {
    // TODO: One problem with this during development: The id of a closure may not change between
    // compilations, which may result in confusion when old values are reported when loaded from a
    // persistent cache even though the closure code was changed in the meantime.
    let id = std::any::TypeId::of::<F>();
    Id::hash(&id)
}
