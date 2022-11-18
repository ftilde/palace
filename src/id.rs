use sha1_smol::Sha1;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Id([u8; 20]);

impl Id {
    pub fn from_data(data: &[u8]) -> Self {
        let sha = Sha1::from(data);
        Self(sha.digest().bytes())
    }
    pub fn combine(ids: &[Id]) -> Self {
        let mut sha = Sha1::new();
        for id in ids {
            sha.update(&id.0);
        }
        Self(sha.digest().bytes())
    }
}
