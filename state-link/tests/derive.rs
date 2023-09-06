#[derive(state_link::State, Default, Debug, PartialEq)]
struct NewtypeArray([f32; 3], u32);

#[test]
fn newtype_array() {
    let mut f = NewtypeArray::default();
    f.0[1] = 0.23;

    let mut store = state_link::Store::default();

    let r = store.store(&f);

    let n = store.load(&r).unwrap();
    let n0 = store.load(&r.elm0()).unwrap();

    assert_eq!(f, n);

    assert_eq!(f.0, n0);
}

#[derive(state_link::State, Default, Debug, PartialEq)]
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

    let n = store.load(&r).unwrap();
    let nb = store.load(&r.b()).unwrap();

    assert_eq!(f, n);

    assert_eq!(f.b, nb);
}

#[derive(state_link::State, Default, Debug, PartialEq)]
struct Unit;

#[test]
fn unit() {
    use state_link::State;

    let f = Unit;

    let mut store = state_link::Store::default();

    let r = f.store(&mut store);

    let n = Unit::load(&store, r).unwrap();

    assert_eq!(f, n);
}

#[derive(state_link::State, Default, Debug, PartialEq)]
struct Generics<T>(T, u32);
#[test]
fn generics() {
    use state_link::State;

    let mut f = Generics::<f32>::default();
    f.0 = 0.23;

    let mut store = state_link::Store::default();

    let r = f.store(&mut store);

    let n = Generics::load(&store, r).unwrap();

    assert_eq!(f, n);
}
