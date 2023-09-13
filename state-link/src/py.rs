use std::any::TypeId;

use pyo3::prelude::*;

#[derive(Default)]
#[pyclass]
pub struct Store {
    inner: super::Store,
}

#[pymethods]
impl Store {
    #[new]
    fn new() -> Self {
        Default::default()
    }

    fn store_u32(&mut self, py: Python, val: u32) -> PyObject {
        let init = NodeHandleU32::build(NodeHandleScalar::new::<u32>(self.inner.store(&val).inner));
        PyCell::new(py, init).unwrap().to_object(py)
    }

    fn store_f32(&mut self, py: Python, val: f32) -> PyObject {
        let init = NodeHandleF32::build(NodeHandleScalar::new::<f32>(self.inner.store(&val).inner));
        PyCell::new(py, init).unwrap().to_object(py)
    }

    fn store_f32_arr(&mut self, py: Python, val: [f32; 3]) -> PyObject {
        let init = NodeHandleArray::new::<f32>(self.inner.store(&val).inner, 3);
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

#[pyclass(subclass)]
#[derive(Clone)]
pub struct NodeHandleScalar {
    type_: TypeId,
    type_name: &'static str,
    inner: super::GenericNodeHandle,
    load: fn(Python, &super::GenericNodeHandle, &super::Store) -> PyObject,
}

impl NodeHandleScalar {
    fn new<T: std::any::Any + IntoPy<PyObject> + super::State>(
        inner: super::GenericNodeHandle,
    ) -> Self {
        Self {
            type_: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            inner,
            load: |py, node, store| store.load_unchecked::<T>(node).into_py(py),
        }
    }
}
#[pymethods]
impl NodeHandleScalar {
    fn link_to(&self, dst: &NodeHandleScalar, store: &mut Store) -> PyResult<()> {
        if self.type_ != dst.type_ {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Argument type mismatch: {} vs {}",
                self.type_name, dst.type_name,
            )));
        }

        store.inner.link_unchecked(&self.inner, &dst.inner);
        Ok(())
    }

    fn load(&self, py: Python, store: &Store) -> PyObject {
        (self.load)(py, &self.inner, &store.inner)
    }
}

#[pyclass(extends = NodeHandleScalar)]
pub struct NodeHandleF32;

#[pymethods]
impl NodeHandleF32 {
    #[new]
    fn build(inner: NodeHandleScalar) -> PyClassInitializer<Self> {
        PyClassInitializer::from(inner).add_subclass(NodeHandleF32)
    }
    fn write(self_: PyRef<'_, Self>, val: f32, store: &mut Store) {
        let super_ = self_.into_super();

        store.inner.write_unchecked(&super_.inner, &val);
    }
}

#[pyclass(extends = NodeHandleScalar)]
pub struct NodeHandleU32;

#[pymethods]
impl NodeHandleU32 {
    #[new]
    fn build(inner: NodeHandleScalar) -> PyClassInitializer<Self> {
        PyClassInitializer::from(inner).add_subclass(NodeHandleU32)
    }
    fn write(self_: PyRef<'_, Self>, val: u32, store: &mut Store) {
        let super_ = self_.into_super();

        store.inner.write_unchecked(&super_.inner, &val);
    }
}

#[pyclass]
pub struct NodeHandleArray {
    item_type_: TypeId,
    item_type_name: &'static str,
    len: usize,
    inner: super::GenericNodeHandle,
    load_item: fn(Python, &super::GenericNodeHandle, &super::Store) -> PyObject,
    write_item: fn(Python, &super::GenericNodeHandle, &PyObject, &mut super::Store) -> PyResult<()>,
    build_item_handle: fn(Python, super::GenericNodeHandle) -> PyObject,
}

fn write_item<T: super::State + Clone + for<'f> FromPyObject<'f>>(
    py: Python,
    node: &super::GenericNodeHandle,
    obj: &PyObject,
    store: &mut super::Store,
) -> PyResult<()> {
    let val = obj.extract::<T>(py)?;
    store.write_unchecked(node, &val);
    Ok(())
}

trait PyState: super::State + Clone + IntoPy<PyObject> + for<'f> FromPyObject<'f> + 'static {
    fn build_handle(py: Python, inner: super::GenericNodeHandle) -> PyObject;
}

impl PyState for f32 {
    fn build_handle(py: Python, inner: super::GenericNodeHandle) -> PyObject {
        let init = NodeHandleF32::build(NodeHandleScalar::new::<Self>(inner));
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

impl PyState for u32 {
    fn build_handle(py: Python, inner: super::GenericNodeHandle) -> PyObject {
        let init = NodeHandleU32::build(NodeHandleScalar::new::<Self>(inner));
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

impl<const I: usize, T: PyState> PyState for [T; I] {
    fn build_handle(py: Python, inner: super::GenericNodeHandle) -> PyObject {
        let init = NodeHandleArray::new::<T>(inner, I);
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

impl NodeHandleArray {
    fn new<
        T: std::any::Any
            + IntoPy<PyObject>
            + super::State
            + Clone
            + for<'f> FromPyObject<'f>
            + PyState,
    >(
        inner: super::GenericNodeHandle,
        len: usize,
    ) -> Self {
        Self {
            item_type_: TypeId::of::<T>(),
            item_type_name: std::any::type_name::<T>(),
            inner,
            len,
            load_item: |py, node, store| store.load_unchecked::<T>(node).into_py(py),
            write_item: write_item::<T>,
            build_item_handle: T::build_handle,
        }
    }
}

#[pymethods]
impl NodeHandleArray {
    fn load(&self, py: Python, store: &Store) -> PyObject {
        let mut out: Vec<PyObject> = Vec::with_capacity(self.len);
        for i in 0..self.len {
            out.push((self.load_item)(py, &self.inner.index(i), &store.inner));
        }

        out.into_py(py)
    }

    fn link_to(&self, dst: &NodeHandleArray, store: &mut Store) -> PyResult<()> {
        if self.item_type_ != dst.item_type_ {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Item type mismatch: {} vs {}",
                self.item_type_name, dst.item_type_name,
            )));
        }

        if self.len != dst.len {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Sequence len mismatch: {} vs {}",
                self.len, dst.len,
            )));
        }

        store.inner.link_unchecked(&self.inner, &dst.inner);
        Ok(())
    }

    fn write(&self, py: Python, vals: Vec<PyObject>, store: &mut Store) -> PyResult<()> {
        let at = self.inner.node;
        let seq = if let super::ResolveResult::Seq(seq) = store.inner.to_val(at) {
            seq.clone() //TODO: instead of cloning we can probably also just take the old value out
        } else {
            panic!("Not a sequence");
        };

        if seq.len() != self.len {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Sequence len mismatch: {} vs {}",
                self.len,
                seq.len(),
            )));
        }

        for (v, slot) in vals.iter().zip(seq.iter()) {
            (self.write_item)(
                py,
                &super::GenericNodeHandle::new_at(*slot),
                v,
                &mut store.inner,
            )?;
        }

        Ok(())
    }

    fn at(&self, py: Python, i: usize) -> PyResult<PyObject> {
        if i >= self.len {
            let message = format!(
                "Index {} out of range for handle of sequence with len {}",
                i, self.len
            );
            return Err(pyo3::exceptions::PyIndexError::new_err(message));
        }
        Ok((self.build_item_handle)(py, self.inner.index(i)))
    }
}

#[pyclass]
pub struct SomeStruct {
    #[pyo3(get, set)]
    v1: u32,
    #[pyo3(get, set)]
    v2: f32,
    #[pyo3(get, set)]
    v3: [f32; 3],
}

impl super::State for SomeStruct {
    type NodeHandle = super::NodeHandleSpecialized<SomeStruct>;

    fn write(&self, store: &mut crate::Store, at: crate::NodeRef) -> crate::Result<()> {
        let map = if let super::ResolveResult::Struct(map) = store.to_val(at) {
            map.clone()
        } else {
            return Err(super::Error::IncorrectType);
        };

        {
            let field_name = "v1";
            let loc = map
                .get(field_name)
                .ok_or(super::Error::MissingField(field_name.to_owned()))?;
            self.v1.write(store, *loc)?;
        }

        {
            let field_name = "v2";
            let loc = map
                .get(field_name)
                .ok_or(super::Error::MissingField(field_name.to_owned()))?;
            self.v2.write(store, *loc)?;
        }

        {
            let field_name = "v3";
            let loc = map
                .get(field_name)
                .ok_or(super::Error::MissingField(field_name.to_owned()))?;
            self.v3.write(store, *loc)?;
        }

        Ok(())
    }

    fn store(&self, store: &mut crate::Store) -> crate::NodeRef {
        let mut map = super::Map::default();
        map.insert("v1".to_owned(), self.v1.store(store));
        map.insert("v2".to_owned(), self.v2.store(store));
        map.insert("v3".to_owned(), self.v3.store(store));

        store.push(super::Node::Dir(map))
    }

    fn load(store: &crate::Store, location: crate::NodeRef) -> crate::Result<Self> {
        if let super::ResolveResult::Struct(map) = store.to_val(location) {
            Ok(Self {
                v1: {
                    let field_name = stringify!(v1);
                    let loc = map
                        .get(field_name)
                        .ok_or(super::Error::MissingField(field_name.to_owned()))?;
                    <u32>::load(store, *loc)?
                },
                v2: {
                    let field_name = stringify!(v2);
                    let loc = map
                        .get(field_name)
                        .ok_or(super::Error::MissingField(field_name.to_owned()))?;
                    <f32>::load(store, *loc)?
                },
                v3: {
                    let field_name = stringify!(v3);
                    let loc = map
                        .get(field_name)
                        .ok_or(super::Error::MissingField(field_name.to_owned()))?;
                    <[f32; 3]>::load(store, *loc)?
                },
            })
        } else {
            Err(super::Error::IncorrectType)
        }
    }
}

#[pymethods]
impl SomeStruct {
    #[new]
    fn new() -> Self {
        SomeStruct {
            v1: 0,
            v2: 0.0,
            v3: [0.0; 3],
        }
    }

    fn store(&self, py: Python, store: &mut Store) -> PyObject {
        let init = NodeHandleSomeStruct::build(NodeHandleScalar::new::<Self>(
            store.inner.store(self).inner,
        ));
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

#[pyclass(extends = NodeHandleScalar)]
pub struct NodeHandleSomeStruct;

#[pymethods]
impl NodeHandleSomeStruct {
    #[new]
    fn build(inner: NodeHandleScalar) -> PyClassInitializer<Self> {
        PyClassInitializer::from(inner).add_subclass(NodeHandleSomeStruct)
    }
    fn write(self_: PyRef<'_, Self>, val: &SomeStruct, store: &mut Store) {
        let super_ = self_.into_super();

        store.inner.write_unchecked(&super_.inner, val);
    }

    fn v1(self_: PyRef<'_, Self>, py: Python) -> PyObject {
        let super_ = self_.into_super();
        u32::build_handle(py, super_.inner.named("v1".to_owned()))
    }

    fn v2(self_: PyRef<'_, Self>, py: Python) -> PyObject {
        let super_ = self_.into_super();
        f32::build_handle(py, super_.inner.named("v2".to_owned()))
    }

    fn v3(self_: PyRef<'_, Self>, py: Python) -> PyObject {
        let super_ = self_.into_super();
        <[f32; 3]>::build_handle(py, super_.inner.named("v3".to_owned()))
    }
}
