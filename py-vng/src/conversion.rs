use pyo3::{FromPyObject, PyAny, PyResult};

// TODO: Investigate just using an enum with FromPyObject instead. That might be quite a lot
// simpler...

pub trait FromPyValue<V>: Sized {
    fn from_py(v: V) -> PyResult<Self>;
}

pub struct ToOperatorFrom<Op, VS>(Op, std::marker::PhantomData<VS>);

pub trait ToOperatorFromT<Op> {
    fn operator(self) -> Op;
}

impl<Op, VS> ToOperatorFromT<Op> for ToOperatorFrom<Op, VS> {
    fn operator(self) -> Op {
        self.0
    }
}

impl<'source, V1: FromPyObject<'source>, Op: FromPyObject<'source> + FromPyValue<V1>>
    FromPyObject<'source> for ToOperatorFrom<Op, (V1,)>
{
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        let v = if let Ok(v) = val.extract::<Op>() {
            v
        } else {
            Op::from_py(val.extract::<V1>()?)?
        };
        Ok(ToOperatorFrom(v, Default::default()))
    }
}

impl<
        'source,
        V1: FromPyObject<'source>,
        V2: FromPyObject<'source>,
        Op: FromPyObject<'source> + FromPyValue<V1> + FromPyValue<V2>,
    > FromPyObject<'source> for ToOperatorFrom<Op, (V1, V2)>
{
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        let v = if let Ok(v) = val.extract::<ToOperatorFrom<Op, (V1,)>>() {
            v.0
        } else {
            Op::from_py(val.extract::<V2>()?)?
        };
        Ok(ToOperatorFrom(v, Default::default()))
    }
}

impl<
        'source,
        V1: FromPyObject<'source>,
        V2: FromPyObject<'source>,
        V3: FromPyObject<'source>,
        Op: FromPyObject<'source> + FromPyValue<V1> + FromPyValue<V2> + FromPyValue<V3>,
    > FromPyObject<'source> for ToOperatorFrom<Op, (V1, V2, V3)>
{
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        let v = if let Ok(v) = val.extract::<ToOperatorFrom<Op, (V1, V2)>>() {
            v.0
        } else {
            Op::from_py(val.extract::<V3>()?)?
        };
        Ok(ToOperatorFrom(v, Default::default()))
    }
}

pub trait FromPyValues<'source>: Sized {
    type Converter: FromPyObject<'source> + ToOperatorFromT<Self>;
}

pub struct ToOperator<Op>(pub Op);

// Welp, this ain't working because of a conflicting implementation for From
//impl<Op> Into<Op> for ToOperator<Op> {
//    fn into(self) -> Op {
//        self.0
//    }
//}

impl<'source, Op: FromPyValues<'source>> FromPyObject<'source> for ToOperator<Op> {
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        Ok(ToOperator(val.extract::<Op::Converter>()?.operator()))
    }
}
