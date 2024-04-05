use id::Identify;

#[derive(Identify)]
struct Struct<'a> {
    a: f32,
    b: &'a str,
}

#[derive(Identify)]
struct TupleStruct(u32, u8);

#[derive(Identify)]
struct UnitStruct;

#[test]
fn r#struct() {
    UnitStruct.id();
    Struct { a: 0.0, b: "bla" }.id();
    TupleStruct(0, 2).id();
}

#[derive(Identify)]
enum Enum<'a> {
    Unit,
    Struct { a: f32, b: &'a str },
    Tuple(u32, u8),
}

#[test]
fn r#enum() {
    Enum::Unit.id();
    Enum::Struct { a: 0.0, b: "bla" }.id();
    Enum::Tuple(0, 2).id();
}
