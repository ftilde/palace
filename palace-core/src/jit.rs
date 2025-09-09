use derive_more::From;
use std::fmt::{Display, Write};

use ash::vk;
use futures::StreamExt;
use id::{Id, Identify};

use crate::{
    array::TensorMetaData,
    dim::{DDyn, DynDimension},
    dtypes::{AsDynType, DType, ElementType, ScalarType},
    op_descriptor,
    operator::{DataParam, DataParamWithExternalId, OperatorDescriptor, OperatorParameter},
    operators::tensor::TensorOperator,
    storage::{
        gpu::{InplaceHandle, InplaceResult, WriteHandle},
        Element,
    },
    task::{Request, RequestStream},
    vec::Vector,
    vulkan::{
        pipeline::{AsDescriptors, ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::{Config, Shader},
        DstBarrierInfo, SrcBarrierInfo,
    },
};

#[derive(id::Identify, Clone, Debug)]
enum NullaryOp {
    Const(ConstValue),
    Read(InputId),
    Position,
    Dimensions,
}

struct WriteNullary(NullaryOp);
impl Display for WriteNullary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            NullaryOp::Const(const_value) => write!(f, "{}", const_value),
            NullaryOp::Read(input_id) => write!(f, "{}", input_id),
            NullaryOp::Position => write!(
                f,
                "add(from_linear(gID, consts.chunk_size), consts.chunk_offset)",
            ),
            NullaryOp::Dimensions => write!(f, "consts.dimensions"),
        }
    }
}

#[derive(id::Identify, Clone, Copy, Debug)]
pub enum FoldOp {
    Sum,
    Mul,
    Min,
    Max,
}

struct WriteFold(FoldOp, NodeId, NodeId);

impl Display for WriteFold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            FoldOp::Sum => write!(f, "{}+{}[{}]", self.1, self.2, FOLD_LOOP_VARIABLE_NAME),
            FoldOp::Mul => write!(f, "{}*{}[{}]", self.1, self.2, FOLD_LOOP_VARIABLE_NAME),
            FoldOp::Min => write!(f, "min({},{}[{}])", self.1, self.2, FOLD_LOOP_VARIABLE_NAME),
            FoldOp::Max => write!(f, "min({},{}[{}])", self.1, self.2, FOLD_LOOP_VARIABLE_NAME),
        }
    }
}

#[derive(id::Identify, Clone, Copy, Debug)]
pub enum UnaryOp {
    Abs,
    Neg,
    Log,
    Exp,
    Sqrt,
    Reinterpret(DType),
    Cast(DType),
    Index(u32),
    IndexRange(u32, u32),
    Splat(u32),
    Fold(FoldOp),
}

impl UnaryOp {
    fn dtype(&self, input: DType) -> Result<DType, crate::Error> {
        Ok(match self {
            UnaryOp::Abs => {
                if input.scalar.is_signed() {
                    input
                } else {
                    return Err(format!(
                        "Value of unsigned type {:?} cannot be passed to abs",
                        input
                    )
                    .into());
                }
            }
            UnaryOp::Log => input,
            UnaryOp::Exp => input,
            UnaryOp::Sqrt => input,
            UnaryOp::Cast(output) => {
                if input.vec_size() == output.vec_size() {
                    *output
                } else {
                    return Err(format!("{:?} cannot be converted to {:?}", input, output).into());
                }
            }
            UnaryOp::Neg => {
                if input.scalar.is_signed() {
                    input
                } else {
                    return Err(
                        format!("Value of unsigned type {:?} cannot be negated", input).into(),
                    );
                }
            }
            UnaryOp::Index(i) => {
                if *i < input.size {
                    input.scalar.into()
                } else {
                    return Err(format!(
                        "Index {} out of range for vector value of size {}",
                        i, input.size
                    )
                    .into());
                }
            }
            UnaryOp::IndexRange(from, to) => {
                if from >= to {
                    return Err(format!(
                        "From ({}) must be strictly smaller that to ({})",
                        from, to
                    )
                    .into());
                }
                if *to <= input.size {
                    input.scalar.vec(to - from).into()
                } else {
                    return Err(format!(
                        "End index {} out of range for vector value of size {}",
                        to, input.size
                    )
                    .into());
                }
            }
            UnaryOp::Splat(size) => {
                if input.size == 1 {
                    DType {
                        scalar: input.scalar,
                        size: *size,
                    }
                } else {
                    return Err(
                        format!("Cannot splat non-scalar value (dim {})", input.size).into(),
                    );
                }
            }
            UnaryOp::Fold(_) => input.scalar.into(),
            UnaryOp::Reinterpret(output) => {
                if input.size != output.size {
                    return Err(format!(
                        "Input ({:?}) and output ({:?}) must have the same vector size",
                        input, output
                    )
                    .into());
                }

                match (input.scalar, output.scalar) {
                    (ScalarType::U32, ScalarType::F32) | (ScalarType::F32, ScalarType::U32) => {
                        *output
                    }
                    _ => {
                        return Err(format!("Cannot reinterpret {:?} as {:?}", input, output).into())
                    }
                }
            }
        })
    }
}
struct WriteUnary(NodeId, UnaryOp, NodeWithDType);
impl Display for WriteUnary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = &self.2;
        match self.1 {
            UnaryOp::Log => write!(f, "log({})", v),
            UnaryOp::Exp => write!(f, "exp({})", v),
            UnaryOp::Abs => write!(f, "abs({})", v),
            UnaryOp::Sqrt => write!(f, "sqrt({})", v),
            UnaryOp::Cast(output) => write!(f, "{}({})", output.scalar.glsl_type(), v),
            UnaryOp::Neg => write!(f, "-{}", v),
            UnaryOp::Index(i) => write!(f, "{}[{}]", v.0, i),
            UnaryOp::IndexRange(from, _to) => {
                if v.1.size == 1 {
                    write!(f, "{}[{}]", v.0, from)
                } else {
                    write!(f, "{}[{}+{}]", v.0, from, VEC_LOOP_VARIABLE_NAME)
                }
            }
            UnaryOp::Splat(_i) => write!(f, "{}", v),
            UnaryOp::Fold(fold_op) => {
                if v.1.size == 1 {
                    write!(f, "{}", v.0)
                } else {
                    let fold = WriteFold(fold_op, self.0, v.0);
                    write!(
                                f,
                                "{input}[0]; for(int {var} = 1; {var} < {size}; ++{var}) {{ {out} = {fold};}}",
                                input = v.0,
                                size = v.1.size,
                                out = self.0,
                                fold = fold,
                                var = FOLD_LOOP_VARIABLE_NAME
                            )
                }
            }
            UnaryOp::Reinterpret(dtype) => match dtype.scalar {
                ScalarType::F32 => write!(f, "uintBitsToFloat({})", v),
                ScalarType::U32 => write!(f, "floatBitsToUint({})", v),
                _ => panic!("Unsupported dtype (should not get here)"),
            },
        }
    }
}

#[derive(id::Identify, Clone, Copy, Debug)]
pub enum BinOp {
    Add,
    Mul,
    Sub,
    Div,
    Max,
    Min,
    GreaterThan,
    GreaterThanEquals,
    LessThan,
    LessThanEquals,
    Equals,
    NotEquals,
    Concat,
}

impl BinOp {
    fn dtype(&self, input1: DType, input2: DType) -> Result<DType, crate::Error> {
        if input1.scalar == input2.scalar {
            match self {
                BinOp::Add | BinOp::Mul | BinOp::Sub | BinOp::Div | BinOp::Max | BinOp::Min
                    if input1.size == input2.size =>
                {
                    Ok(input1)
                }
                BinOp::GreaterThan
                | BinOp::GreaterThanEquals
                | BinOp::LessThan
                | BinOp::LessThanEquals
                | BinOp::Equals
                | BinOp::NotEquals
                    if input1.size == input2.size =>
                {
                    Ok(ScalarType::U32.into())
                }
                BinOp::Concat => Ok({
                    let mut out = input1;
                    out.size += input2.size;
                    out
                }),
                _ => Err(format!(
                    "Mismatched dtype sizes {:?} and {:?} for binary op",
                    input1.size, input2.size
                )
                .into()),
            }
        } else {
            Err(format!(
                "Mismatched dtypes {:?} and {:?} for binary op",
                input1, input2
            )
            .into())
        }
    }
}

#[derive(id::Identify, Clone, Copy, Debug)]
pub enum TernaryOp {
    IfThenElse,
}

impl TernaryOp {
    fn dtype(&self, _input1: DType, input2: DType, input3: DType) -> Result<DType, crate::Error> {
        match self {
            TernaryOp::IfThenElse => {
                if input2 == input3 {
                    Ok(input2)
                } else {
                    Err(format!(
                        "DTypes of 2nd ({}) and 3nd ({})argument must match",
                        input2, input3
                    )
                    .into())
                }
            }
        }
    }
}

const VEC_LOOP_VARIABLE_NAME: &'static str = "i";
const FOLD_LOOP_VARIABLE_NAME: &'static str = "j";
struct VecLoop(u32);
impl Display for VecLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 1 {
            Ok(())
        } else {
            write!(f, "for(int i=0; i<{}; ++i)", self.0)
        }
    }
}

struct WriteValue(NodeId, u32);
impl Display for WriteValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1 == 1 {
            write!(f, "{}", self.0)
        } else {
            write!(f, "{}[{}]", self.0, VEC_LOOP_VARIABLE_NAME)
        }
    }
}

#[derive(Copy, Clone)]
struct NodeWithDType(NodeId, DType);
impl Display for NodeWithDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.1.size;
        if s == 1 {
            write!(f, "{}", self.0)
        } else {
            write!(f, "{}[{}]", self.0, VEC_LOOP_VARIABLE_NAME)
        }
    }
}

struct WriteBin(BinOp, NodeWithDType, NodeWithDType);
impl Display for WriteBin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let l = &self.1;
        let r = &self.2;
        match self.0 {
            BinOp::Add => write!(f, "{} + {}", l, r),
            BinOp::Sub => write!(f, "{} - {}", l, r),
            BinOp::Mul => write!(f, "{} * {}", l, r),
            BinOp::Div => write!(f, "{} / {}", l, r),
            BinOp::Max => write!(f, "max({}, {})", l, r),
            BinOp::Min => write!(f, "min({}, {})", l, r),
            BinOp::GreaterThan => write!(f, "uint({} > {})", l, r),
            BinOp::GreaterThanEquals => write!(f, "uint({} >= {})", l, r),
            BinOp::LessThan => write!(f, "uint({} < {})", l, r),
            BinOp::LessThanEquals => write!(f, "uint({} <= {})", l, r),
            BinOp::Equals => write!(f, "uint({} == {})", l, r),
            BinOp::NotEquals => write!(f, "uint({} != {})", l, r),
            BinOp::Concat => {
                write!(f, "({} < {}) ? {} : ", VEC_LOOP_VARIABLE_NAME, l.1.size, l)?;
                if r.1.size == 1 {
                    write!(f, "{}", r.0)
                } else {
                    write!(f, "{}[{} - {}]", r.0, VEC_LOOP_VARIABLE_NAME, l.1.size)
                }
            }
        }
    }
}

struct WriteTernary(TernaryOp, NodeWithDType, NodeWithDType, NodeWithDType);
impl Display for WriteTernary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a0 = &self.1;
        let a1 = &self.2;
        let a2 = &self.3;

        match self.0 {
            TernaryOp::IfThenElse => write!(f, "({} != 0) ? {} : {}", a0, a1, a2),
        }
    }
}

#[derive(id::Identify, Clone, Debug, From)]
pub enum ConstValueScalar {
    F32(f32),
    U32(u32),
}

impl Into<ConstValue> for ConstValueScalar {
    fn into(self) -> ConstValue {
        match self {
            ConstValueScalar::F32(v) => ConstValue::F32(Vector::new(vec![v])),
            ConstValueScalar::U32(v) => ConstValue::U32(Vector::new(vec![v])),
        }
    }
}

#[derive(id::Identify, Clone, Debug, From)]
pub enum ConstValue {
    F32(Vector<DDyn, f32>),
    U32(Vector<DDyn, u32>),
}

fn write_vec<T: Display + Element + AsDynType>(
    f: &mut std::fmt::Formatter<'_>,
    v: &Vector<DDyn, T>,
) -> std::fmt::Result {
    let len = v.len();
    let mut t: DType = T::D_TYPE;
    t.size = len as _;
    write!(f, "{}(", t.glsl_type())?;
    for i in 0..len {
        if i != 0 {
            write!(f, ",")?;
        }
        write!(f, "{}", v[i])?;
    }
    write!(f, ")")
}

impl Display for ConstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstValue::F32(v) => write_vec(f, v),
            ConstValue::U32(v) => write_vec(f, v),
        }
    }
}

impl ConstValue {
    fn dtype(&self) -> DType {
        match self {
            ConstValue::F32(v) => ScalarType::F32.vec(v.len() as _),
            ConstValue::U32(v) => ScalarType::U32.vec(v.len() as _),
        }
    }
}

#[derive(id::Identify, Clone, Debug)]
enum Instruction {
    NullAry(DType, NullaryOp),
    Unary(DType, UnaryOp, InstructionOffset),
    Binary(DType, BinOp, InstructionOffset, InstructionOffset),
    Ternary(
        DType,
        TernaryOp,
        InstructionOffset,
        InstructionOffset,
        InstructionOffset,
    ),
}

impl Instruction {
    fn dtype(&self) -> DType {
        match self {
            Instruction::NullAry(dtype, ..) => *dtype,
            Instruction::Unary(dtype, ..) => *dtype,
            Instruction::Binary(dtype, ..) => *dtype,
            Instruction::Ternary(dtype, ..) => *dtype,
        }
    }
}

#[derive(Clone, Identify)]
struct OrderedSet<T: Identify>(Vec<T>);

impl<T: Identify> OrderedSet<T> {
    fn add(&mut self, item: T) -> usize {
        let pos = if let Some(pos) = self
            .0
            .iter()
            .enumerate()
            .find_map(|(i, n)| (n.id() == item.id()).then_some(i))
        {
            pos
        } else {
            let pos = self.0.len();
            self.0.push(item);
            pos
        };
        pos
    }
}

//TODO: Rc for cheap clone?
#[derive(Clone)]
pub struct JitTensorOperator<D: DynDimension> {
    metadata: Option<TensorMetaData<D>>,
    dtype: DType,
    operators: OrderedSet<TensorOperator<D, DType>>,
    instructions: Vec<Instruction>,
    id: Id,
}
impl<D: DynDimension> Identify for JitTensorOperator<D> {
    fn id(&self) -> Id {
        self.id
    }
}

fn merge_instructions<D: DynDimension>(
    operators0: OrderedSet<TensorOperator<D, DType>>,
    instructions0: Vec<Instruction>,
    operators1: OrderedSet<TensorOperator<D, DType>>,
    instructions1: Vec<Instruction>,
) -> (
    OrderedSet<TensorOperator<D, DType>>,
    Vec<Instruction>,
    usize,
    usize,
) {
    let mut inputs = operators0;
    let old_inputs = operators1;

    let mut ops = instructions0;
    let mut new_ops = instructions1;
    for op in &mut new_ops {
        if let Instruction::NullAry(_, NullaryOp::Read(input_id)) = op {
            *input_id = InputId(inputs.add(old_inputs.0[input_id.0].clone()));
        }
    }

    let offset_0 = new_ops.len();
    let offset_1 = 0;

    ops.extend(new_ops);

    (inputs, ops, offset_0, offset_1)
}

fn merge_metadata<D: DynDimension>(
    md0: Option<TensorMetaData<D>>,
    md1: Option<TensorMetaData<D>>,
) -> Result<Option<TensorMetaData<D>>, crate::Error> {
    Ok(match (md0, md1) {
        (None, None) => None,
        (None, r) => r,
        (l, None) => l,
        (l, r) => {
            if l == r {
                l
            } else {
                return Err(format!("Metadata mismatch {:?}, {:?}", l, r).into());
            }
        }
    })
}

impl<D: DynDimension> OperatorParameter for JitTensorOperator<D> {
    fn data_longevity(&self) -> crate::storage::DataLongevity {
        self.operators
            .0
            .iter()
            .map(|o| o.data_longevity())
            .min()
            .unwrap_or(crate::storage::DataLongevity::Stable)
    }
}

impl<D: DynDimension> JitTensorOperator<D> {
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn metadata(&self) -> Option<TensorMetaData<D>> {
        self.metadata.clone()
    }
    pub fn with_md(mut self, new_md: TensorMetaData<D>) -> Result<Self, crate::Error> {
        self.metadata = if let Some(md) = self.metadata {
            if new_md == md {
                Some(md)
            } else {
                return Err(
                    format!("Mismatch with existing metadata {:?}, {:?}", md, new_md,).into(),
                );
            }
        } else {
            Some(new_md)
        };
        Ok(self)
    }
    pub fn unary_op(op: UnaryOp, inner: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        if let UnaryOp::Cast(target) = &op {
            if *target == inner.dtype {
                return Ok(inner);
            }
        }

        Ok({
            let id = Id::combine(&[op.id(), inner.id()]);
            let dtype = op.dtype(inner.dtype)?;
            let op = Instruction::Unary(dtype, op, InstructionOffset(1));
            let mut ops = inner.instructions;
            ops.push(op);
            Self {
                metadata: inner.metadata.clone(),
                dtype,
                operators: inner.operators,
                instructions: ops,
                id,
            }
        })
    }
    pub fn bin_op(
        op: BinOp,
        l: JitTensorOperator<D>,
        r: JitTensorOperator<D>,
    ) -> Result<Self, crate::Error> {
        Ok({
            let id = Id::combine(&[op.id(), l.id(), r.id()]);
            let dtype = op.dtype(l.dtype, r.dtype)?;

            let (inputs, mut ops, offset_l, offset_r) =
                merge_instructions(l.operators, l.instructions, r.operators, r.instructions);

            let op = Instruction::Binary(
                dtype,
                op,
                InstructionOffset(offset_l + 1),
                InstructionOffset(offset_r + 1),
            );
            ops.push(op);

            Self {
                metadata: merge_metadata(l.metadata, r.metadata)?,
                dtype,
                operators: inputs,
                instructions: ops,
                id,
            }
        })
    }
    pub fn ternary_op(
        op: TernaryOp,
        a0: JitTensorOperator<D>,
        a1: JitTensorOperator<D>,
        a2: JitTensorOperator<D>,
    ) -> Result<Self, crate::Error> {
        Ok({
            let id = Id::combine(&[op.id(), a0.id(), a1.id(), a2.id()]);
            let dtype = op.dtype(a0.dtype, a1.dtype, a2.dtype)?;

            let (inputs_initial, ops_initial, offset_0_initial, _) =
                merge_instructions(a0.operators, a0.instructions, a1.operators, a1.instructions);

            let (inputs, mut ops, offset_1, offset_2) =
                merge_instructions(inputs_initial, ops_initial, a2.operators, a2.instructions);

            let offset_0 = offset_0_initial + offset_1;

            let op = Instruction::Ternary(
                dtype,
                op,
                InstructionOffset(offset_0 + 1),
                InstructionOffset(offset_1 + 1),
                InstructionOffset(offset_2 + 1),
            );
            ops.push(op);

            let metadata = merge_metadata(merge_metadata(a0.metadata, a1.metadata)?, a2.metadata)?;

            Self {
                metadata,
                dtype,
                operators: inputs,
                instructions: ops,
                id,
            }
        })
    }
}

impl<D: DynDimension> From<ConstValue> for JitTensorOperator<D> {
    fn from(c: ConstValue) -> Self {
        let dtype = c.dtype();
        let op = Instruction::NullAry(c.dtype(), NullaryOp::Const(c));
        let id = op.id();
        let ops = vec![op];
        Self {
            instructions: ops,
            metadata: None,
            dtype,
            operators: OrderedSet(Vec::new()),
            id,
        }
    }
}

impl<D: DynDimension> From<f32> for JitTensorOperator<D> {
    fn from(value: f32) -> Self {
        ConstValue::F32(Vector::new(vec![value])).into()
    }
}

pub fn scalar<D: DynDimension, V: Into<ConstValueScalar>>(value: V) -> JitTensorOperator<D> {
    let c: ConstValueScalar = value.into();
    let c: ConstValue = c.into();
    c.into()
}

pub fn const_vec<D: DynDimension, V: Into<ConstValue>>(value: V) -> JitTensorOperator<D> {
    let c: ConstValue = value.into();
    c.into()
}

pub fn dimensions<D: DynDimension>(d: D) -> JitTensorOperator<D> {
    let dtype = ScalarType::U32.vec(d.n() as _);
    let op = Instruction::NullAry(dtype, NullaryOp::Dimensions);
    let id = op.id();
    let ops = vec![op];
    JitTensorOperator {
        instructions: ops,
        metadata: None,
        dtype,
        operators: OrderedSet(Vec::new()),
        id,
    }
}

pub fn position<D: DynDimension>(d: D) -> JitTensorOperator<D> {
    let dtype = ScalarType::U32.vec(d.n() as _);
    let op = Instruction::NullAry(dtype, NullaryOp::Position);
    let id = op.id();
    let ops = vec![op];
    JitTensorOperator {
        instructions: ops,
        metadata: None,
        dtype,
        operators: OrderedSet(Vec::new()),
        id,
    }
}

impl<D: DynDimension> From<TensorOperator<D, DType>> for JitTensorOperator<D> {
    fn from(c: TensorOperator<D, DType>) -> Self {
        let dtype = c.chunks.dtype();
        let metadata = Some(c.metadata.clone());

        let op = Instruction::NullAry(dtype, NullaryOp::Read(InputId(0)));
        let id = Id::combine(&[c.id(), op.id()]);
        let ops = vec![op];

        Self {
            instructions: ops,
            metadata,
            dtype,
            operators: OrderedSet(vec![c]),
            id,
        }
    }
}

impl<D: DynDimension> JitTensorOperator<D> {
    pub fn add(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Add, self, other)
    }
    pub fn sub(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Sub, self, other)
    }
    pub fn mul(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Mul, self, other)
    }
    pub fn div(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Div, self, other)
    }
    pub fn max(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Max, self, other)
    }
    pub fn min(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Min, self, other)
    }
    pub fn concat(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Concat, self, other)
    }
    pub fn and(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Mul, self, other)
    }
    pub fn or(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Add, self, other)
    }

    pub fn square(self) -> Self {
        self.clone().mul(self).unwrap()
    }
    pub fn abs(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Abs, self)
    }
    pub fn neg(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Neg, self)
    }
    pub fn log(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Log, self)
    }
    pub fn exp(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Exp, self)
    }
    pub fn sqrt(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Sqrt, self)
    }
    pub fn cast(self, to: DType) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Cast(to), self)
    }
    pub fn reinterpret(self, to: DType) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Reinterpret(to), self)
    }
    pub fn splat(self, size: u32) -> Self {
        Self::unary_op(UnaryOp::Splat(size), self).unwrap()
    }
    pub fn index(self, i: u32) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Index(i), self)
    }
    pub fn index_range(self, from: u32, to: u32) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::IndexRange(from, to), self)
    }
    pub fn hmul(self) -> Self {
        Self::unary_op(UnaryOp::Fold(FoldOp::Mul), self).unwrap()
    }
    pub fn hadd(self) -> Self {
        Self::unary_op(UnaryOp::Fold(FoldOp::Sum), self).unwrap()
    }
    pub fn hmin(self) -> Self {
        Self::unary_op(UnaryOp::Fold(FoldOp::Min), self).unwrap()
    }
    pub fn hmax(self) -> Self {
        Self::unary_op(UnaryOp::Fold(FoldOp::Max), self).unwrap()
    }

    pub fn select(
        self,
        yes: JitTensorOperator<D>,
        no: JitTensorOperator<D>,
    ) -> Result<Self, crate::Error> {
        Self::ternary_op(TernaryOp::IfThenElse, self, yes, no)
    }

    pub fn lt(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::LessThan, self, other)
    }

    pub fn lt_eq(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::LessThanEquals, self, other)
    }

    pub fn gt(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::GreaterThan, self, other)
    }

    pub fn gt_eq(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::GreaterThanEquals, self, other)
    }

    pub fn eq(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Equals, self, other)
    }

    pub fn neq(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::NotEquals, self, other)
    }
}

#[derive(id::Identify, Copy, Clone, Debug)]
struct InputId(usize);

impl Display for InputId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input{}", self.0)
    }
}

#[derive(Copy, Clone)]
struct NodeId(usize);

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(id::Identify, Copy, Clone, Debug)]
struct InstructionOffset(usize);

pub fn jit<D: DynDimension>(op: TensorOperator<D, DType>) -> JitTensorOperator<D> {
    op.into()
}

fn compile(
    instructions: &Vec<Instruction>,
    input_types: &Vec<DType>,
) -> Result<(String, Config), crate::Error> {
    let mut shader = String::new();
    let mut config = Config::new().ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT));

    writeln!(&mut shader, "#include<size_util.glsl>")?;
    writeln!(&mut shader, "#include<vec.glsl>")?;

    for (i, dtype) in input_types.iter().enumerate() {
        writeln!(
            &mut shader,
            r#"
            layout(std430, binding = {}) readonly buffer InputBuffer{}{{
                {} values[BRICK_MEM_SIZE];
            }} input{};
            "#,
            i,
            i,
            dtype.glsl_type(),
            i,
        )?;
    }

    writeln!(
        &mut shader,
        r#"
            layout(std430, binding = {}) buffer OutputBuffer{{
                {} values[BRICK_MEM_SIZE];
            }} output_buf;
            "#,
        input_types.len(),
        instructions.last().unwrap().dtype().glsl_type(),
    )?;

    writeln!(
        &mut shader,
        r#"
            declare_push_consts(consts);
            void main() {{
                uint gID = global_position_linear;

                if(gID < BRICK_MEM_SIZE) {{
            "#
    )?;

    for (i, dtype) in input_types.iter().enumerate() {
        writeln!(
            &mut shader,
            "{} {} = input{}.values[gID];",
            dtype.glsl_type(),
            InputId(i),
            i
        )?;
    }

    #[derive(Debug)]
    struct Node<'a> {
        instr: &'a Instruction,
        id: Id,
        instruction_number: usize,
    }

    impl Identify for Node<'_> {
        fn id(&self) -> Id {
            self.id
        }
    }

    // Perform a collapse-of-identical-instructions optimization step:
    // Calculate an id for all instructions and, based on that collect them in a new list
    // ("nodes"), which automatically deduplicates.

    let mut nodes: OrderedSet<Node> = OrderedSet(Vec::new());
    let mut instruction_to_node: Vec<NodeWithDType> = Vec::new();

    for (i, instr) in instructions.iter().enumerate() {
        let id = match instr {
            Instruction::NullAry(dtype, op) => Id::combine(&[dtype.id(), op.id()]),
            Instruction::Unary(dtype, unary_op, offset) => {
                let node_num = instruction_to_node[i - offset.0];
                Id::combine(&[dtype.id(), unary_op.id(), nodes.0[node_num.0 .0].id])
            }
            Instruction::Binary(dtype, bin_op, offset_l, offset_r) => {
                let node_num_l = instruction_to_node[i - offset_l.0];
                let node_num_r = instruction_to_node[i - offset_r.0];
                Id::combine(&[
                    dtype.id(),
                    bin_op.id(),
                    nodes.0[node_num_l.0 .0].id,
                    nodes.0[node_num_r.0 .0].id,
                ])
            }
            Instruction::Ternary(dtype, bin_op, offset_0, offset_1, offset_2) => {
                let node_num_0 = instruction_to_node[i - offset_0.0];
                let node_num_1 = instruction_to_node[i - offset_1.0];
                let node_num_2 = instruction_to_node[i - offset_2.0];
                Id::combine(&[
                    dtype.id(),
                    bin_op.id(),
                    nodes.0[node_num_0.0 .0].id,
                    nodes.0[node_num_1.0 .0].id,
                    nodes.0[node_num_2.0 .0].id,
                ])
            }
        };

        let node_id = nodes.add(Node {
            instr: &instr,
            id,
            instruction_number: i,
        });

        let node_id = NodeId(node_id);
        let dtype = instr.dtype();
        instruction_to_node.push(NodeWithDType(node_id, dtype));
    }

    let input_node_id = |own_instruction_number: usize, input_offset: InstructionOffset| {
        let input_instruction_number = own_instruction_number - input_offset.0;
        instruction_to_node[input_instruction_number]
    };

    for (i, node) in nodes.0.iter().enumerate() {
        let res_id = NodeId(i);
        let dtype = match node.instr {
            Instruction::NullAry(t, c) => {
                writeln!(
                    &mut shader,
                    "{} {} = {};",
                    t.glsl_type(),
                    res_id,
                    WriteNullary(c.clone())
                )?;
                *t
            }
            Instruction::Unary(t, o, a) => {
                writeln!(
                    &mut shader,
                    "{} {}; {} {} = {};",
                    t.glsl_type(),
                    res_id,
                    VecLoop(t.size),
                    WriteValue(res_id, t.size),
                    WriteUnary(res_id, *o, input_node_id(node.instruction_number, *a))
                )?;
                *t
            }
            Instruction::Binary(t, o, l, r) => {
                writeln!(
                    &mut shader,
                    "{} {}; {} {} = {};",
                    t.glsl_type(),
                    res_id,
                    VecLoop(t.size),
                    WriteValue(res_id, t.size),
                    WriteBin(
                        *o,
                        input_node_id(node.instruction_number, *l),
                        input_node_id(node.instruction_number, *r),
                    )
                )?;
                *t
            }
            Instruction::Ternary(t, o, a0, a1, a2) => {
                writeln!(
                    &mut shader,
                    "{} {}; {} {} = {};",
                    t.glsl_type(),
                    res_id,
                    VecLoop(t.size),
                    WriteValue(res_id, t.size),
                    WriteTernary(
                        *o,
                        input_node_id(node.instruction_number, *a0),
                        input_node_id(node.instruction_number, *a1),
                        input_node_id(node.instruction_number, *a2),
                    )
                )?;
                *t
            }
        };
        config = config.ext(dtype.glsl_ext());
    }

    let root_node_id = NodeId(nodes.0.len() - 1);

    writeln!(
        &mut shader,
        r#"
                    output_buf.values[gID] = {};
                }}
            }}
            "#,
        root_node_id,
    )?;

    Ok((shader, config))
}
impl<D: DynDimension> JitTensorOperator<D> {
    pub fn compile(mut self) -> Result<TensorOperator<D, DType>, crate::Error> {
        enum MaybeInplaceResult<'a, 'inv> {
            Inplace(InplaceResult<'a, 'inv>),
            NotInplace,
        }
        enum MaybeInplaceHandle<'a> {
            Inplace(InplaceHandle<'a>),
            NotInplace(WriteHandle<'a>),
        }

        let Some(metadata) = self.metadata.clone() else {
            return Err("No metadata information in JitOperator".into());
        };

        if let &[Instruction::NullAry(_, NullaryOp::Read(_))] = self.instructions.as_slice() {
            assert_eq!(self.operators.0.len(), 1);
            return Ok(self.operators.0.pop().unwrap());
        }

        let nd = metadata.dim().n();
        let push_constants = DynPushConstants::new()
            .vec::<u32>(nd, "dimensions")
            .vec::<u32>(nd, "chunk_size")
            .vec::<u32>(nd, "chunk_offset");

        Ok(TensorOperator::with_state(
            op_descriptor!(),
            self.dtype,
            metadata.clone(),
            (self, DataParam(metadata), DataParam(push_constants)),
            |ctx, positions, loc, (jit_operator, metadata, push_constants)| {
                async move {
                    let inplace_operator_index = jit_operator.operators.0.iter().position(|o| {
                        o.dtype().element_layout() == jit_operator.dtype.element_layout()
                    });

                    let device = ctx.preferred_device(loc);

                    let access_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    };
                    let m = (*metadata).clone().into_dyn();

                    let num_chunk_elements = m.num_chunk_elements();

                    let input_dtypes = jit_operator
                        .operators
                        .0
                        .iter()
                        .map(|op| op.dtype())
                        .collect::<Vec<_>>();
                    let pipeline = device.request_state(
                        (
                            &input_dtypes,
                            DataParamWithExternalId(&jit_operator.instructions, jit_operator.id),
                            num_chunk_elements,
                            push_constants,
                        ),
                        |device, (input_dtypes, instructions, num_chunk_elements, push_constants)| {
                            let (shader, config) = compile(*instructions, input_dtypes)?;
                            //println!("{}", shader.as_str());
                            ComputePipelineBuilder::new(
                                Shader::new(shader.as_str())
                                    .push_const_block_dyn(push_constants)
                                    .define("BRICK_MEM_SIZE", num_chunk_elements)
                                    .with_config(config),
                            )
                            .use_push_descriptor(true)
                            .build(device)
                        },
                    )?;

                    let mut brick_stream = ctx
                        .submit_unordered_with_data(positions.iter().map(|pos| {
                            (
                                ctx.group(
                                    jit_operator
                                        .operators
                                        .0
                                        .iter()
                                        .enumerate()
                                        .filter(|(i, _)| Some(*i) != inplace_operator_index)
                                        .map(|(_, input)| {
                                            input.chunks.request_gpu(device.id, *pos, access_info)
                                        }),
                                ),
                                *pos,
                            )
                        }))
                        .then_req_with_data(*ctx, |(input, pos)| {
                            if let Some(inplace_operator_index) = inplace_operator_index {
                                let output = jit_operator.operators.0[inplace_operator_index]
                                    .chunks
                                    .request_inplace_gpu(
                                        device.id,
                                        pos,
                                        ctx.current_op_desc().unwrap(),
                                        jit_operator.dtype,
                                        num_chunk_elements,
                                        access_info,
                                    )
                                    .map(|v| MaybeInplaceResult::Inplace(v));
                                (output, (input, pos))
                            } else {
                                //let output = ctx
                                //    .alloc_slot_gpu(device, pos, num_chunk_elements)
                                //    .map(|v| MaybeInplaceResult::NotInplace(v));
                                let output = Request::ready(MaybeInplaceResult::NotInplace);
                                (output, (input, pos))
                            }
                        })
                        .then_req_with_data(*ctx, |(output, (inputs, pos))| {
                            let output = match output {
                                MaybeInplaceResult::Inplace(v) => {
                                    v.alloc().map(|v| MaybeInplaceHandle::Inplace(v))
                                }
                                MaybeInplaceResult::NotInplace => ctx
                                    .alloc_slot_gpu(device, pos, &m.chunk_size)
                                    .map(|v| MaybeInplaceHandle::NotInplace(v)),
                            };
                            (output, (inputs, pos))
                        });

                    while let Some((output, (inputs, pos))) = brick_stream.next().await {
                        device.with_cmd_buffer(|cmd| {
                            let mut descriptors: Vec<&dyn AsDescriptors> = Vec::new();
                            match &output {
                                MaybeInplaceHandle::Inplace(h) => {
                                    let i = inplace_operator_index.unwrap();
                                    descriptors.extend(
                                        inputs[..i].iter().map(|v| v as &dyn AsDescriptors),
                                    );
                                    match h {
                                        InplaceHandle::Inplace(rw, _) => descriptors.push(rw),
                                        InplaceHandle::New(r, _) => descriptors.push(r),
                                    }
                                    descriptors.extend(
                                        inputs[i..].iter().map(|v| v as &dyn AsDescriptors),
                                    );
                                    match h {
                                        InplaceHandle::Inplace(rw, _) => descriptors.push(rw),
                                        InplaceHandle::New(_, w) => descriptors.push(w),
                                    }
                                }
                                MaybeInplaceHandle::NotInplace(w) => {
                                    descriptors
                                        .extend(inputs.iter().map(|v| v as &dyn AsDescriptors));
                                    descriptors.push(w);
                                }
                            };

                            let descriptor_config = DescriptorConfig::from_vec(descriptors);

                            let global_size = num_chunk_elements;

                            let chunk_info = metadata.chunk_info(pos);

                            unsafe {
                                let has_push_consts = pipeline.has_push_constants();

                                let mut pipeline = pipeline.bind(cmd);

                                if has_push_consts {
                                    pipeline.push_constant_dyn(&push_constants, |w| {
                                        w.vec(&metadata.dimensions.raw())?;
                                        w.vec(&metadata.chunk_size.raw())?;
                                        w.vec(&chunk_info.begin().raw())?;

                                        Ok(())
                                    });
                                }

                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(device, global_size);
                            }
                        });

                        unsafe {
                            match output {
                                MaybeInplaceHandle::Inplace(h) => h.initialized(
                                    *ctx,
                                    SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_WRITE,
                                    },
                                ),
                                MaybeInplaceHandle::NotInplace(w) => w.initialized(
                                    *ctx,
                                    SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_WRITE,
                                    },
                                ),
                            }
                        };
                    }

                    Ok(())
                }
                .into()
            },
        ))
    }
}

impl<D: DynDimension> TryInto<TensorOperator<D, DType>> for JitTensorOperator<D> {
    type Error = crate::Error;
    fn try_into(self) -> Result<TensorOperator<D, DType>, Self::Error> {
        self.compile()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        test_util::compare_tensor,
        vec::{LocalVoxelPosition, VoxelPosition},
    };

    #[test]
    fn two_inputs() {
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let input_fn_1 = |v: VoxelPosition| (v.x().raw * v.y().raw * v.z().raw) as f32;
        let input_fn_2 = |v: VoxelPosition| (v.x().raw + v.y().raw + v.z().raw) as f32;

        let input1 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_1).into());
        let input2 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_2).into());

        let output = input1
            .cast(ScalarType::I32.into())
            .unwrap()
            .neg()
            .unwrap()
            .add(
                input2
                    .clone()
                    .cast(ScalarType::I32.into())
                    .unwrap()
                    .mul(input2.cast(ScalarType::I32.into()).unwrap())
                    .unwrap(),
            )
            .unwrap()
            .cast(ScalarType::F32.into())
            .unwrap()
            .compile()
            .unwrap();

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let input1 = input_fn_1(v);
            let input2 = input_fn_2(v);

            (-(input1 as i32) + (input2 as i32) * (input2 as i32)) as f32
        });

        compare_tensor(output.try_into().unwrap(), expected);
    }

    #[test]
    fn asymmetric() {
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let input_fn_1 = |v: VoxelPosition| (v.x().raw * v.y().raw * v.z().raw) as f32;
        let input_fn_2 = |v: VoxelPosition| (v.x().raw + v.y().raw + v.z().raw) as f32;

        let input1 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_1).into());
        let input2 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_2).into());

        let output = input1.sub(input2).unwrap().compile().unwrap();

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let input1 = input_fn_1(v);
            let input2 = input_fn_2(v);

            input1 - input2
        });

        compare_tensor(output.try_into().unwrap(), expected);
    }

    #[test]
    fn ternary() {
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let input_fn_1 = |v: VoxelPosition| (v.x().raw > 2) as i32 as f32;
        let input_fn_2 = |v: VoxelPosition| (v.x().raw * v.y().raw * v.z().raw) as f32;
        let input_fn_3 = |v: VoxelPosition| (v.x().raw + v.y().raw + v.z().raw) as f32;

        let input1 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_1).into());
        let input2 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_2).into());
        let input3 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_3).into());

        let output = input1
            .neq(0.0.into())
            .unwrap()
            .select(input2, input3)
            .unwrap()
            .compile()
            .unwrap();

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let input1 = input_fn_1(v);
            let input2 = input_fn_2(v);
            let input3 = input_fn_3(v);

            if input1 != 0.0 {
                input2
            } else {
                input3
            }
        });

        compare_tensor(output.try_into().unwrap(), expected);
    }

    #[test]
    fn concat() {
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let input_fn_1 = |v: VoxelPosition| (v.x().raw > 2) as i32 as f32;
        let input_fn_2 = |v: VoxelPosition| (v.x().raw * v.y().raw * v.z().raw) as f32;

        let input1 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_1).into());
        let input2 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_2).into());

        let output = input1.concat(input2).unwrap();
        let output1 = output.clone().index(0).unwrap().compile().unwrap();
        let output2 = output.index(1).unwrap().compile().unwrap();

        let expected1 =
            crate::operators::rasterize_function::voxel(size, brick_size, move |v| input_fn_1(v));
        let expected2 =
            crate::operators::rasterize_function::voxel(size, brick_size, move |v| input_fn_2(v));

        compare_tensor(output1.try_into().unwrap(), expected1);
        compare_tensor(output2.try_into().unwrap(), expected2);
    }

    #[test]
    fn dimensions() {
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let f = move |_v: VoxelPosition| size.hmul() as f32;

        let output = super::dimensions(size.dim())
            .with_md(TensorMetaData {
                dimensions: size,
                chunk_size: brick_size,
            })
            .unwrap()
            .hmul()
            .cast(ScalarType::F32.into())
            .unwrap()
            .compile()
            .unwrap();

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| f(v));

        compare_tensor(output.try_into().unwrap(), expected);
    }

    #[test]
    fn position() {
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let f = move |v: VoxelPosition| v.raw().hadd() as f32;

        let output = super::position(size.dim())
            .with_md(TensorMetaData {
                dimensions: size,
                chunk_size: brick_size,
            })
            .unwrap()
            .hadd()
            .cast(ScalarType::F32.into())
            .unwrap()
            .compile()
            .unwrap();

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| f(v));

        compare_tensor(output.try_into().unwrap(), expected);
    }

    #[test]
    fn reinterpret() {
        let s = [1, 1, 1];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);
        let val = 0xffff_abcd;

        let output = super::scalar(ConstValueScalar::U32(val))
            .with_md(TensorMetaData {
                dimensions: size,
                chunk_size: brick_size,
            })
            .unwrap()
            .reinterpret(ScalarType::F32.into())
            .unwrap()
            .compile()
            .unwrap();

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |_| {
            f32::from_bits(val)
        });

        compare_tensor(output.try_into().unwrap(), expected);
    }
}
