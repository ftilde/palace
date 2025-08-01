use crate::types::*;
use palace_core::jit::{BinOp, JitTensorOperator, TernaryOp, UnaryOp};
use palace_core::{dim::*, jit as cjit};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_type_union_enum;

#[gen_stub_type_union_enum]
#[derive(FromPyObject, Clone)]
pub enum JitArgument {
    Tensor(MaybeEmbeddedTensorOperatorArg),
    Const(f32),
}

impl JitArgument {
    pub fn into_jit(self) -> cjit::JitTensorOperator<DDyn> {
        match self {
            JitArgument::Tensor(t) => t.unpack().into_inner().into_jit(),
            JitArgument::Const(c) => c.into(),
        }
    }
}

#[pyfunction]
pub fn jit(c: f32) -> TensorOperator {
    let op: cjit::JitTensorOperator<DDyn> = c.into();
    op.into()
}

pub fn jit_unary(op: UnaryOp, v1: &TensorOperator) -> PyResult<TensorOperator> {
    let core = v1.clone().into_jit();
    let res = crate::map_result(JitTensorOperator::<DDyn>::unary_op(op, core).into())?;
    Ok(res.into())
}

pub fn jit_binary(op: BinOp, v1: &TensorOperator, v2: JitArgument) -> PyResult<TensorOperator> {
    let core = v1.clone().into_jit();
    let res = crate::map_result(JitTensorOperator::<DDyn>::bin_op(op, core, v2.into_jit()).into())?;
    Ok(res.into())
}

pub fn jit_ternary(
    op: TernaryOp,
    v1: &TensorOperator,
    v2: JitArgument,
    v3: JitArgument,
) -> PyResult<TensorOperator> {
    let core = v1.clone().into_jit();
    let res = crate::map_result(
        JitTensorOperator::<DDyn>::ternary_op(op, core, v2.into_jit(), v3.into_jit()).into(),
    )?;
    Ok(res.into())
}
