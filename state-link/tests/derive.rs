#[derive(state_link::StateNoPy, Default, Debug, PartialEq)]
struct NewtypeArray([f32; 3], u32);

#[test]
fn newtype_array() {
    let mut f = NewtypeArray::default();
    f.0[1] = 0.23;

    let mut store = state_link::Store::default();

    let r = store.store(&f);

    let n = store.load(&r);
    let n0 = store.load(&r.elm0());

    assert_eq!(f, n);

    assert_eq!(f.0, n0);
}

#[derive(state_link::StateNoPy, Default, Debug, PartialEq)]
struct SimpleStruct {
    a: f32,
    b: f32,
    another: u32,
}

#[test]
fn simple_struct() {
    let mut f = SimpleStruct::default();
    f.b = 0.23;

    let mut store = state_link::Store::default();

    let r = store.store(&f);

    let n = store.load(&r);
    let nb = store.load(&r.b());

    assert_eq!(f, n);

    assert_eq!(f.b, nb);
}

#[derive(state_link::StateNoPy, Default, Debug, PartialEq)]
struct Unit;

#[test]
fn unit() {
    let f = Unit;

    let mut store = state_link::Store::default();

    let r = store.store(&f);

    let n = store.load(&r);

    assert_eq!(f, n);
}

#[derive(state_link::StateNoPy, Debug, PartialEq)]
enum Enum {
    One,
    Two,
}

#[test]
fn derive_enum() {
    let f = Enum::One;

    let mut store = state_link::Store::default();

    let r = store.store(&f);

    let n = store.load(&r);

    assert_eq!(f, n);
}

#[derive(state_link::StateNoPy, Default, Debug, PartialEq)]
struct Generics<T>(T, u32);
#[test]
fn generics() {
    let mut f = Generics::<f32>::default();
    f.0 = 0.23;

    let mut store = state_link::Store::default();

    let r = store.store(&f);

    let n = store.load(&r);

    assert_eq!(f, n);
}

#[test]
fn link_simple() {
    let mut store = state_link::Store::default();

    let f = Generics::<u32>::default();

    let r = store.store(&f);

    store.link(&r.elm0(), &r.elm1()).unwrap();

    store.write(&r.elm0(), &123).unwrap();

    assert_eq!(store.load(&r.elm0()), 123);
    assert_eq!(store.load(&r.elm1()), 123);
}

#[test]
fn link_complex() {
    let mut store = state_link::Store::default();

    let a = Generics::<SimpleStruct>::default();
    let b = NewtypeArray::default();

    let a = store.store(&a);
    let b = store.store(&b);

    store.link(&a.elm0().a(), &b.elm0().at(2)).unwrap();

    store.write(&b.elm0(), &[1.0, 2.0, 3.0]).unwrap();

    assert_eq!(
        store.load(&a.elm0()),
        SimpleStruct {
            a: 3.0,
            ..Default::default()
        }
    );
}
