use sha1_smol::Sha1;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
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
