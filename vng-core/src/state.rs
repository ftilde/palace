use egui::epaint::ahash::HashMap;

use crate::data::Vector;

#[derive(Copy, Clone, Debug)]
struct NodeRef {
    index: usize,
}

#[derive(Default, Debug)]
struct Store {
    elms: Vec<Node>,
}

#[derive(Debug)]
enum Node {
    Dir(HashMap<String, NodeRef>),
    Seq(Vec<NodeRef>),
    Val(Value),
    Link(NodeRef),
}

enum ResolveResult<'a> {
    Struct(&'a HashMap<String, NodeRef>),
    Seq(&'a Vec<NodeRef>),
    Atom(Value),
}

impl Store {
    fn push(&mut self, val: Node) -> NodeRef {
        let index = self.elms.len();
        self.elms.push(val);
        NodeRef { index }
    }

    fn store<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> NodeRef {
        v.serialize(self).unwrap()
    }

    fn load<'a, T: serde::Deserialize<'a> + ?Sized>(&'a self, node: NodeRef) -> Result<T, SDError> {
        T::deserialize(Deserializer { node, store: self })
    }

    fn walk(&self, root: NodeRef, p: &[PathElm]) -> Option<NodeRef> {
        match p {
            [] => Some(root),
            [current, rest @ ..] => match &self.elms[root.index] {
                Node::Dir(d) => {
                    let e = d.get(current.str()?)?;
                    self.walk(*e, rest)
                }
                Node::Seq(s) => {
                    let e = s.get(current.index()?)?;
                    self.walk(*e, rest)
                }
                Node::Val(_) => None,
                Node::Link(l) => self.walk(*l, p),
            },
        }
    }
    fn to_val(&self, root: NodeRef) -> ResolveResult {
        match &self.elms[root.index] {
            Node::Dir(s) => ResolveResult::Struct(&s),
            Node::Seq(s) => ResolveResult::Seq(&s),
            Node::Val(v) => ResolveResult::Atom(*v),
            Node::Link(l) => self.to_val(*l),
        }
    }
}

#[derive(derive_more::From)]
enum PathElm {
    Named(String),
    Index(usize),
}

impl PathElm {
    fn str(&self) -> Option<&str> {
        if let PathElm::Named(s) = self {
            Some(s)
        } else {
            None
        }
    }
    fn index(&self) -> Option<usize> {
        if let PathElm::Index(i) = self {
            Some(*i)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Value {
    F32(f32),
    U32(u32),
    Unit,
}

#[derive(Debug)]
struct SDError(String);

impl std::fmt::Display for SDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SDError {}

impl serde::ser::Error for SDError {
    fn custom<T>(msg: T) -> Self
    where
        T: std::fmt::Display,
    {
        Self(msg.to_string())
    }
}

impl serde::de::Error for SDError {
    fn custom<T>(msg: T) -> Self
    where
        T: std::fmt::Display,
    {
        Self(msg.to_string())
    }
}

struct SerializeSeq<'a> {
    store: &'a mut Store,
    vals: Vec<NodeRef>,
}

impl serde::ser::SerializeSeq for SerializeSeq<'_> {
    type Ok = NodeRef;

    type Error = SDError;

    fn serialize_element<T: ?Sized>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: serde::Serialize,
    {
        self.vals.push(self.store.store(value));
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(self.store.push(Node::Seq(self.vals)))
    }
}

struct SerializeStruct<'a> {
    store: &'a mut Store,
    vals: HashMap<String, NodeRef>,
}

impl serde::ser::SerializeStruct for SerializeStruct<'_> {
    type Ok = NodeRef;

    type Error = SDError;

    fn serialize_field<T: ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error>
    where
        T: serde::Serialize,
    {
        self.vals.insert(key.to_owned(), self.store.store(value));
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(self.store.push(Node::Dir(self.vals)))
    }
}

impl<'a> serde::Serializer for &'a mut Store {
    type Ok = NodeRef;

    type Error = SDError;

    type SerializeSeq = SerializeSeq<'a>;

    type SerializeTuple = serde::ser::Impossible<Self::Ok, Self::Error>;

    type SerializeTupleStruct = serde::ser::Impossible<Self::Ok, Self::Error>;

    type SerializeTupleVariant = serde::ser::Impossible<Self::Ok, Self::Error>;

    type SerializeMap = serde::ser::Impossible<Self::Ok, Self::Error>;

    type SerializeStruct = SerializeStruct<'a>;

    type SerializeStructVariant = serde::ser::Impossible<Self::Ok, Self::Error>;

    fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i8(self, _v: i8) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i16(self, _v: i16) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i32(self, _v: i32) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i64(self, _v: i64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u8(self, _v: u8) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u16(self, _v: u16) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        Ok(self.push(Node::Val(Value::U32(v))))
    }

    fn serialize_u64(self, _v: u64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        Ok(self.push(Node::Val(Value::F32(v))))
    }

    fn serialize_f64(self, _v: f64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_str(self, _v: &str) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_some<T: ?Sized>(self, _value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: serde::Serialize,
    {
        todo!()
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Ok(self.push(Node::Val(Value::Unit)))
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_newtype_struct<T: ?Sized>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: serde::Serialize,
    {
        todo!()
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: serde::Serialize,
    {
        todo!()
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Ok(SerializeSeq {
            store: self,
            vals: Vec::new(),
        })
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        todo!()
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        todo!()
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        __variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        todo!()
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        todo!()
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(SerializeStruct {
            store: self,
            vals: HashMap::default(),
        })
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        todo!()
    }
}

struct Deserializer<'a> {
    store: &'a Store,
    node: NodeRef,
}

struct SeqAccess<'a> {
    store: &'a Store,
    nodes: &'a [NodeRef],
}

impl<'de> serde::de::SeqAccess<'de> for SeqAccess<'de> {
    type Error = SDError;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: serde::de::DeserializeSeed<'de>,
    {
        match self.nodes {
            [] => Ok(None),
            [head, tail @ ..] => {
                self.nodes = tail;
                seed.deserialize(Deserializer {
                    store: self.store,
                    node: *head,
                })
                .map(Some)
            }
        }
    }
}

struct MapAccess<'a, I> {
    store: &'a Store,
    nodes: I,
    trailing_value: Option<NodeRef>,
}

impl<'de, I: Iterator<Item = (&'de String, &'de NodeRef)>> serde::de::MapAccess<'de>
    for MapAccess<'de, I>
{
    type Error = SDError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: serde::de::DeserializeSeed<'de>,
    {
        if let Some((k, v)) = self.nodes.next() {
            self.trailing_value = Some(*v);
            seed.deserialize(serde::de::value::StrDeserializer::new(k))
                .map(Some)
        } else {
            Ok(None)
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::DeserializeSeed<'de>,
    {
        let node = self.trailing_value.unwrap();
        seed.deserialize(Deserializer {
            store: self.store,
            node,
        })
    }
}

impl<'de> serde::Deserializer<'de> for Deserializer<'de> {
    type Error = SDError;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_bool<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i8<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i16<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i32<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_i64<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_u8<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_u16<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        if let ResolveResult::Atom(Value::U32(u)) = self.store.to_val(self.node) {
            visitor.visit_u32(u)
        } else {
            Err(SDError("Not a u32".to_owned()))
        }
    }

    fn deserialize_u64<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        if let ResolveResult::Atom(Value::F32(u)) = self.store.to_val(self.node) {
            visitor.visit_f32(u)
        } else {
            Err(SDError("Not a f32".to_owned()))
        }
    }

    fn deserialize_f64<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_char<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_str<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_string<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_bytes<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_byte_buf<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_option<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        if let ResolveResult::Atom(Value::Unit) = self.store.to_val(self.node) {
            visitor.visit_unit()
        } else {
            Err(SDError("Not a unit".to_owned()))
        }
    }

    fn deserialize_unit_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        if let ResolveResult::Seq(s) = self.store.to_val(self.node) {
            visitor.visit_seq(SeqAccess {
                store: self.store,
                nodes: &s[..],
            })
        } else {
            Err(SDError("Not a seq".to_owned()))
        }
    }

    fn deserialize_tuple<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        if let ResolveResult::Struct(s) = self.store.to_val(self.node) {
            visitor.visit_map(MapAccess {
                store: self.store,
                nodes: s.into_iter(),
                trailing_value: None,
            })
        } else {
            Err(SDError("Not a map/struct".to_owned()))
        }
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_identifier<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }

    fn deserialize_ignored_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn f32() {
        let mut store = Store::default();
        let v = 12.0f32;
        let i = store.store(&v);

        let r = store.load::<f32>(i).unwrap();

        assert_eq!(v, r);
    }

    #[test]
    fn vec() {
        let mut store = Store::default();
        let v = Vector::from([2.0, 3.1, 0.55]);
        let i = store.store(&v);

        let r = store.load::<Vector<3, f32>>(i).unwrap();

        assert_eq!(v, r);
    }

    #[test]
    fn md() {
        let mut store = Store::default();
        let v = crate::array::ImageMetaData {
            dimensions: Vector::from([2, 5]),
            chunk_size: Vector::from([1, 2]),
        };
        let i = store.store(&v);

        let r = store.load::<crate::array::ImageMetaData>(i).unwrap();

        assert_eq!(v, r);
    }
}
