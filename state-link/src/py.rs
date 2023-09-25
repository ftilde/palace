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

    //fn store_u32(&mut self, py: Python, val: u32) -> PyObject {
    //    let init = NodeHandleU32 {
    //        inner: self.inner.store(&val),
    //    };
    //    PyCell::new(py, init).unwrap().to_object(py)
    //}

    //fn store_f32(&mut self, py: Python, val: f32) -> PyObject {
    //    let init = NodeHandleF32 {
    //        inner: self.inner.store(&val),
    //    };
    //    PyCell::new(py, init).unwrap().to_object(py)
    //}

    //fn store_f32_arr(self, py: Python, val: [f32; 3]) -> PyObject {
    //    let store = Py::new(self)
    //    let init = NodeHandleArray::new::<f32>(self.inner.store(&val).inner, 3, store);
    //    PyCell::new(py, init).unwrap().to_object(py)
    //}
}

#[pyclass]
pub struct NodeHandleF32 {
    inner: <f32 as super::State>::NodeHandle,
    store: Py<Store>,
}

#[pymethods]
impl NodeHandleF32 {
    fn write(&self, py: Python, val: f32) -> PyResult<()> {
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
    fn load(&self, py: Python) -> PyObject {
        self.store
            .borrow_mut(py)
            .inner
            .load(&self.inner)
            .into_py(py)
    }

    fn map(&self, py: pyo3::Python, f: &pyo3::types::PyFunction) -> pyo3::PyResult<()> {
        let val_py = self.load(py);
        let res_py = f.call1((&val_py,))?;
        let val = res_py.extract::<f32>()?;
        self.write(py, val)
    }
}

#[pyclass]
pub struct NodeHandleU32 {
    inner: <u32 as super::State>::NodeHandle,
    store: Py<Store>,
}

#[pymethods]
impl NodeHandleU32 {
    fn write(&self, py: Python, val: u32) -> PyResult<()> {
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
    fn load(&self, py: Python) -> PyObject {
        self.store
            .borrow_mut(py)
            .inner
            .load(&self.inner)
            .into_py(py)
    }
    fn map(&self, py: pyo3::Python, f: &pyo3::types::PyFunction) -> pyo3::PyResult<()> {
        let val_py = self.load(py);
        let res_py = f.call1((&val_py,))?;
        let val = res_py.extract::<u32>()?;
        self.write(py, val)
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
    build_item_handle: fn(Python, super::GenericNodeHandle, Py<Store>) -> PyObject,
    store: Py<Store>,
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
    fn build_handle(py: Python, inner: super::GenericNodeHandle, store: Py<Store>) -> PyObject;
}

impl PyState for f32 {
    fn build_handle(py: Python, inner: super::GenericNodeHandle, store: Py<Store>) -> PyObject {
        let init = NodeHandleF32 {
            inner: <<Self as super::State>::NodeHandle as super::NodeHandle>::pack(inner),
            store,
        };
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

impl PyState for u32 {
    fn build_handle(py: Python, inner: super::GenericNodeHandle, store: Py<Store>) -> PyObject {
        let init = NodeHandleU32 {
            inner: <<Self as super::State>::NodeHandle as super::NodeHandle>::pack(inner),
            store,
        };
        PyCell::new(py, init).unwrap().to_object(py)
    }
}

impl<const I: usize, T: PyState> PyState for [T; I] {
    fn build_handle(py: Python, inner: super::GenericNodeHandle, store: Py<Store>) -> PyObject {
        let init = NodeHandleArray::new::<T>(inner, I, store);
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
        store: Py<Store>,
    ) -> Self {
        Self {
            item_type_: TypeId::of::<T>(),
            item_type_name: std::any::type_name::<T>(),
            inner,
            len,
            load_item: |py, node, store| store.load_unchecked::<T>(node).into_py(py),
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

#[pymethods]
impl NodeHandleArray {
    fn load(&self, py: Python) -> PyObject {
        let store = self.store.borrow(py);
        let mut out: Vec<PyObject> = Vec::with_capacity(self.len);
        for i in 0..self.len {
            out.push((self.load_item)(py, &self.inner.index(i), &store.inner));
        }

        out.into_py(py)
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

    fn mutate(&self, py: Python, f: &pyo3::types::PyFunction) -> PyResult<()> {
        let initial = self.load(py);
        let new_py = f.call1((initial,))?;
        let new = new_py.extract::<Vec<PyObject>>()?;
        self.write(py, new)
    }

    fn map(&self, py: pyo3::Python, f: &pyo3::types::PyFunction) -> pyo3::PyResult<()> {
        let val_py = self.load(py);
        let res_py = f.call1((&val_py,))?;
        let val = res_py.extract::<Vec<PyObject>>()?;
        self.write(py, val)
    }
}
