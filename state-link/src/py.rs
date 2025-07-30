use std::any::TypeId;

use pyo3::{prelude::*, IntoPyObjectExt};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pymethods};

#[pyclass]
#[gen_stub_pyclass]
pub struct Store {
    pub inner: Py<StoreInner>,
}

#[derive(Default)]
#[pyclass]
#[gen_stub_pyclass]
pub struct StoreInner {
    pub inner: super::Store,
}

#[derive(FromPyObject)]
#[gen_stub_pyclass_complex_enum]
enum StorePrimitive {
    U32(u32),
    F32(f32),
    String(String),
}

#[pymethods]
#[gen_stub_pymethods]
impl Store {
    #[new]
    fn new(py: Python) -> Self {
        Self {
            inner: Py::new(py, StoreInner::default()).unwrap(),
        }
    }

    fn store_primitive(&self, py: Python, primitive: StorePrimitive) -> PyObject {
        match primitive {
            StorePrimitive::F32(v) => NodeHandleF32 {
                inner: {
                    let x = self.inner.borrow_mut(py).inner.store(&v);
                    x
                },
                store: self.inner.clone().into(),
            }
            .into_py_any(py),
            StorePrimitive::U32(v) => NodeHandleU32 {
                inner: {
                    let x = self.inner.borrow_mut(py).inner.store(&v);
                    x
                },
                store: self.inner.clone().into(),
            }
            .into_py_any(py),
            StorePrimitive::String(v) => NodeHandleString {
                inner: {
                    let x = self.inner.borrow_mut(py).inner.store(&v);
                    x
                },
                store: self.inner.clone().into(),
            }
            .into_py_any(py),
        }
        .unwrap()
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct NodeHandleF32 {
    inner: <f32 as super::State>::NodeHandle,
    store: Py<StoreInner>,
}

#[gen_stub_pymethods]
#[pymethods]
impl NodeHandleF32 {
    pub fn write(&self, py: Python, val: f32) -> PyResult<()> {
        self.store
            .borrow_mut(py)
            .inner
            .write(&self.inner, &val)
            .map_err(map_link_err)
    }
    fn link_to(&self, py: Python, dst: &NodeHandleF32) -> PyResult<()> {
        self.store
            .borrow_mut(py)
            .inner
            .link(&self.inner, &dst.inner)
            .map_err(map_link_err)
    }
    pub fn load(&self, py: Python) -> f32 {
        self.store.borrow_mut(py).inner.load(&self.inner)
    }

    fn map(&self, py: pyo3::Python, f: PyObject) -> pyo3::PyResult<()> {
        let val_py = self.load(py).into_py_any(py).unwrap();
        let res_py = f.call1(py, (&val_py,))?;
        let val = res_py.extract::<f32>(py)?;
        self.write(py, val)
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct NodeHandleU32 {
    inner: <u32 as super::State>::NodeHandle,
    store: Py<StoreInner>,
}

#[gen_stub_pymethods]
#[pymethods]
impl NodeHandleU32 {
    pub fn write(&self, py: Python, val: u32) -> PyResult<()> {
        self.store
            .borrow_mut(py)
            .inner
            .write(&self.inner, &val)
            .map_err(map_link_err)
    }
    fn link_to(&self, py: Python, dst: &NodeHandleU32) -> PyResult<()> {
        self.store
            .borrow_mut(py)
            .inner
            .link(&self.inner, &dst.inner)
            .map_err(map_link_err)
    }
    pub fn load(&self, py: Python) -> u32 {
        self.store.borrow_mut(py).inner.load(&self.inner)
    }
    fn map(&self, py: pyo3::Python, f: PyObject) -> pyo3::PyResult<()> {
        let val_py = self.load(py).into_py_any(py).unwrap();
        let res_py = f.call1(py, (&val_py,))?;
        let val = res_py.extract::<u32>(py)?;
        self.write(py, val)
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct NodeHandleString {
    pub inner: <String as super::State>::NodeHandle,
    pub store: Py<StoreInner>,
}

#[gen_stub_pymethods]
#[pymethods]
impl NodeHandleString {
    pub fn write(&self, py: Python, val: String) -> PyResult<()> {
        self.store
            .borrow_mut(py)
            .inner
            .write(&self.inner, &val)
            .map_err(map_link_err)
    }
    fn link_to(&self, py: Python, dst: &NodeHandleString) -> PyResult<()> {
        self.store
            .borrow_mut(py)
            .inner
            .link(&self.inner, &dst.inner)
            .map_err(map_link_err)
    }
    pub fn load(&self, py: Python) -> String {
        self.store.borrow_mut(py).inner.load(&self.inner)
    }
    fn map(&self, py: pyo3::Python, f: PyObject) -> pyo3::PyResult<()> {
        let val_py = self.load(py).into_py_any(py).unwrap();
        let res_py = f.call1(py, (&val_py,))?;
        let val = res_py.extract::<String>(py)?;
        self.write(py, val)
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct NodeHandleArray {
    item_type_: TypeId,
    item_type_name: &'static str,
    len: usize,
    inner: super::GenericNodeHandle,
    load_item: fn(Python, &super::GenericNodeHandle, &super::Store) -> PyObject,
    write_item: fn(Python, &super::GenericNodeHandle, &PyObject, &mut super::Store) -> PyResult<()>,
    build_item_handle: fn(Python, super::GenericNodeHandle, Py<StoreInner>) -> PyObject,
    store: Py<StoreInner>,
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
    std::any::Any + super::State + Clone + for<'f> IntoPyObject<'f> + for<'f> FromPyObject<'f> + 'static
{
    fn build_handle(py: Python, inner: super::GenericNodeHandle, store: Py<StoreInner>)
        -> PyObject;
}

impl PyState for f32 {
    fn build_handle(
        py: Python,
        inner: super::GenericNodeHandle,
        store: Py<StoreInner>,
    ) -> PyObject {
        let init = NodeHandleF32 {
            inner: <<Self as super::State>::NodeHandle as super::NodeHandle>::pack(inner),
            store,
        };
        init.into_py_any(py).unwrap()
    }
}

impl PyState for u32 {
    fn build_handle(
        py: Python,
        inner: super::GenericNodeHandle,
        store: Py<StoreInner>,
    ) -> PyObject {
        let init = NodeHandleU32 {
            inner: <<Self as super::State>::NodeHandle as super::NodeHandle>::pack(inner),
            store,
        };
        init.into_py_any(py).unwrap()
    }
}

impl PyState for String {
    fn build_handle(
        py: Python,
        inner: super::GenericNodeHandle,
        store: Py<StoreInner>,
    ) -> PyObject {
        let init = NodeHandleString {
            inner: <<Self as super::State>::NodeHandle as super::NodeHandle>::pack(inner),
            store,
        };
        init.into_py_any(py).unwrap()
    }
}

impl<const I: usize, T: PyState> PyState for [T; I] {
    fn build_handle(
        py: Python,
        inner: super::GenericNodeHandle,
        store: Py<StoreInner>,
    ) -> PyObject {
        let init = NodeHandleArray::new::<T>(inner, I, store);
        init.into_py_any(py).unwrap()
    }
}

impl NodeHandleArray {
    pub fn new<
        T: std::any::Any
            + super::State
            + Clone
            + for<'f> IntoPyObject<'f>
            + for<'f> FromPyObject<'f>
            + PyState,
    >(
        inner: super::GenericNodeHandle,
        len: usize,
        store: Py<StoreInner>,
    ) -> Self {
        Self {
            item_type_: TypeId::of::<T>(),
            item_type_name: std::any::type_name::<T>(),
            inner,
            len,
            load_item: |py, node, store| store.load_unchecked::<T>(node).into_py_any(py).unwrap(),
            write_item: write_item::<T>,
            build_item_handle: T::build_handle,
            store,
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

#[gen_stub_pymethods]
#[pymethods]
impl NodeHandleArray {
    fn load(&self, py: Python) -> PyObject {
        let store = self.store.borrow(py);
        let mut out: Vec<PyObject> = Vec::with_capacity(self.len);
        for i in 0..self.len {
            out.push((self.load_item)(py, &self.inner.index(i), &store.inner));
        }

        out.into_py_any(py).unwrap()
    }

    fn link_to(&self, py: Python, dst: &NodeHandleArray) -> PyResult<()> {
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

        let mut store = self.store.borrow_mut(py);
        store
            .inner
            .link_unchecked(&self.inner, &dst.inner)
            .map_err(map_link_err)?;
        Ok(())
    }

    fn write(&self, py: Python, vals: Vec<PyObject>) -> PyResult<()> {
        let mut store = self.store.borrow_mut(py);

        let at = store.inner.resolve(&self.inner).unwrap();
        let seq =
            if let super::ResolveResult::Seq(seq) = store.inner.to_val(at).map_err(map_link_err)? {
                seq.clone()
            } else {
                panic!("Not a sequence");
            };

        assert_eq!(self.len, seq.len());

        if seq.len() != vals.len() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Sequence len mismatch: {} vs {}",
                seq.len(),
                vals.len(),
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
        Ok((self.build_item_handle)(
            py,
            self.inner.index(i),
            self.store.clone_ref(py),
        ))
    }

    fn mutate(&self, py: Python, f: PyObject) -> PyResult<()> {
        let initial = self.load(py);
        let new_py = f.call1(py, (initial,))?;
        let new = new_py.extract::<Vec<PyObject>>(py)?;
        self.write(py, new)
    }

    fn map(&self, py: pyo3::Python, f: PyObject) -> pyo3::PyResult<()> {
        let val_py = self.load(py);
        let res_py = f.call1(py, (&val_py,))?;
        let val = res_py.extract::<Vec<PyObject>>(py)?;
        self.write(py, val)
    }
}
