use std::any::TypeId;

use pyo3::prelude::*;

#[derive(Default)]
#[pyclass]
pub struct Store {
    pub inner: super::Store,
}

#[pymethods]
impl Store {
    #[new]
    fn new() -> Self {
        Default::default()
    }

    fn store_u32(&mut self, py: Python, val: u32) -> PyObject {
        let init = NodeHandleU32 {
            inner: self.inner.store(&val),
        };
        PyCell::new(py, init).unwrap().to_object(py)
    }

    fn store_f32(&mut self, py: Python, val: f32) -> PyObject {
        let init = NodeHandleF32 {
            inner: self.inner.store(&val),
        };
        PyCell::new(py, init).unwrap().to_object(py)
    }

    fn store_f32_arr(&mut self, py: Python, val: [f32; 3]) -> PyObject {
        let init = NodeHandleArray::new::<f32>(self.inner.store(&val).inner, 3);
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

#[pyclass]
pub struct NodeHandleF32 {
    inner: <f32 as super::State>::NodeHandle,
}

#[pymethods]
impl NodeHandleF32 {
    fn write(&self, val: f32, store: &mut Store) -> PyResult<()> {
        store.inner.write(&self.inner, &val).map_err(map_link_err)
    }
    fn link_to(&self, dst: &NodeHandleF32, store: &mut Store) -> PyResult<()> {
        store
            .inner
            .link(&self.inner, &dst.inner)
            .map_err(map_link_err)
    }
    fn load(&self, py: Python, store: &Store) -> PyObject {
        store.inner.load(&self.inner).into_py(py)
    }
}

#[pyclass]
pub struct NodeHandleU32 {
    inner: <u32 as super::State>::NodeHandle,
}

#[pymethods]
impl NodeHandleU32 {
    fn write(&self, val: u32, store: &mut Store) -> PyResult<()> {
        store.inner.write(&self.inner, &val).map_err(map_link_err)
    }
    fn link_to(&self, dst: &NodeHandleU32, store: &mut Store) -> PyResult<()> {
        store
            .inner
            .link(&self.inner, &dst.inner)
            .map_err(map_link_err)
    }
    fn load(&self, py: Python, store: &Store) -> PyObject {
        store.inner.load(&self.inner).into_py(py)
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
    store.write_unchecked(node, &val).map_err(map_link_err)
}

pub trait PyState:
    std::any::Any + super::State + Clone + IntoPy<PyObject> + for<'f> FromPyObject<'f> + 'static
{
    fn build_handle(py: Python, inner: super::GenericNodeHandle) -> PyObject;
}

impl PyState for f32 {
    fn build_handle(py: Python, inner: super::GenericNodeHandle) -> PyObject {
        let init = NodeHandleF32 {
            inner: <<Self as super::State>::NodeHandle as super::NodeHandle>::pack(inner),
        };
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

impl PyState for u32 {
    fn build_handle(py: Python, inner: super::GenericNodeHandle) -> PyObject {
        let init = NodeHandleU32 {
            inner: <<Self as super::State>::NodeHandle as super::NodeHandle>::pack(inner),
        };
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

pub fn map_link_err(err: super::Error) -> PyErr {
    match err {
        crate::Error::LinkReferenceCycle => {
            pyo3::exceptions::PyValueError::new_err("Detected a link reference cycle.")
        }
        e => panic!("Unexpected error: {:?}", e),
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

        store
            .inner
            .link_unchecked(&self.inner, &dst.inner)
            .map_err(map_link_err)?;
        Ok(())
    }

    fn write(&self, py: Python, vals: Vec<PyObject>, store: &mut Store) -> PyResult<()> {
        let at = self.inner.node;
        let seq =
            if let super::ResolveResult::Seq(seq) = store.inner.to_val(at).map_err(map_link_err)? {
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
