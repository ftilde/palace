use bytemuck::AnyBitPattern;

use crate::{
    data::VoxelPosition,
    storage::{ReadHandle, WriteHandleUninit},
    Error,
};
use std::{
    fs::File,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    path::PathBuf,
};

use crate::{
    data::{to_linear, BrickPosition, VolumeMetaData},
    id::Id,
    task::{Task, TaskContext},
};

impl From<Id> for OperatorId {
    fn from(inner: Id) -> Self {
        Self(inner)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct DataId(Id);
impl DataId {
    pub fn new(op: OperatorId, descriptor: &impl Hash) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        descriptor.hash(&mut hasher);
        let v = hasher.finish();
        let hash = bytemuck::bytes_of(&v);
        let data_id = Id::from_data(hash);

        DataId(Id::combine(&[op.inner(), data_id]))
    }
}

struct TypeErased(*mut ());
impl TypeErased {
    fn pack<T>(v: T) -> Self {
        TypeErased(Box::into_raw(Box::new(v)) as *mut ())
    }
    unsafe fn unpack<T>(self) -> T {
        *Box::from_raw(self.0 as *mut T)
    }
}

trait OpaqueOperator<'op> {
    fn id(&self) -> &OperatorId;
    unsafe fn compute<'tasks>(
        &'tasks self,
        context: TaskContext<'tasks, 'op>,
        items: Vec<TypeErased>,
    ) -> Task<'tasks>;
}

type ResultPoll<'req, 'op, V> = Box<dyn FnMut(TaskContext<'req, 'op>) -> Option<V>>;

pub enum RequestType<'op> {
    Data(DataRequest<'op>),
    ThreadPoolJob(()),
}

pub struct Request<'req, 'op, V> {
    pub id: DataId,
    pub type_: RequestType<'op>,
    pub poll: ResultPoll<'req, 'op, V>,
    pub _marker: std::marker::PhantomData<&'req ()>,
}

type ComputeFunction<'op, ItemDescriptor> =
    Box<dyn for<'tasks> Fn(TaskContext<'tasks, 'op>, Vec<ItemDescriptor>) -> Task<'tasks> + 'op>;

struct Operator<'op, ItemDescriptor, Output: ?Sized> {
    id: OperatorId,
    compute: ComputeFunction<'op, ItemDescriptor>,
    _marker: std::marker::PhantomData<(ItemDescriptor, Output)>,
}

impl<'op, ItemDescriptor: Hash, Output: AnyBitPattern> Operator<'op, ItemDescriptor, Output> {
    fn new<
        F: for<'tasks> Fn(TaskContext<'tasks, 'op>, Vec<ItemDescriptor>) -> Task<'tasks> + 'op,
    >(
        id: OperatorId,
        compute: F,
    ) -> Self {
        Self {
            id,
            compute: Box::new(compute),
            _marker: Default::default(),
        }
    }

    fn request<'req>(
        &'op self,
        item: ItemDescriptor,
    ) -> Request<'req, 'op, ReadHandle<'req, [Output]>> {
        Request {
            id: DataId::new(self.id, &item),
            type_: RequestType::Data(DataRequest {
                source: self,
                item: TypeErased::pack(item),
            }),
            poll: Box::new(move |ctx| unsafe {
                ctx.storage.read_ram_slice(/*self.id*/ todo!())
            }),
            _marker: Default::default(),
        }
    }
}

impl<'op, ItemDescriptor, Output> OpaqueOperator<'op> for Operator<'op, ItemDescriptor, Output> {
    fn id(&self) -> &OperatorId {
        &self.id
    }
    unsafe fn compute<'tasks>(
        &'tasks self,
        context: TaskContext<'tasks, 'op>,
        items: Vec<TypeErased>,
    ) -> Task<'tasks> {
        let items = items
            .into_iter()
            .map(|v| unsafe { v.unpack() })
            .collect::<Vec<_>>();
        (self.compute)(context, items)
    }
}

pub struct DataRequest<'op> {
    source: &'op dyn OpaqueOperator<'op>,
    item: TypeErased,
}

pub type ScalarOperator<'op, T> = Operator<'op, (), T>;

impl<'op, T: bytemuck::Pod> From<&'op T> for ScalarOperator<'op, T> {
    fn from(value: &'op T) -> Self {
        scalar(value)
    }
}

fn scalar<'op, T: bytemuck::Pod>(val: &'op T) -> ScalarOperator<'op, T> {
    let op_id = OperatorId::new(
        std::any::type_name::<T>(),
        &[Id::from_data(bytemuck::bytes_of(val)).into()],
    );
    Operator::new(op_id, move |ctx, d| {
        async move {
            let id = DataId::new(op_id, &d);
            ctx.storage.write_to_ram(todo!() /*id*/, *val)
        }
        .into()
    })
}

struct VolumeOperator<'op> {
    metadata: Operator<'op, (), VolumeMetaData>,
    bricks: Operator<'op, BrickPosition, f32>,
}

impl<'op> VolumeOperator<'op> {
    fn new<
        M: for<'tasks> Fn(TaskContext<'tasks, 'op>, Vec<()>) -> Task<'tasks> + 'op,
        B: for<'tasks> Fn(TaskContext<'tasks, 'op>, Vec<BrickPosition>) -> Task<'tasks> + 'op,
    >(
        base_id: OperatorId,
        metadata: M,
        bricks: B,
    ) -> Self {
        Self {
            metadata: Operator::new(base_id.slot(0), metadata),
            bricks: Operator::new(base_id.slot(1), bricks),
        }
    }
}

fn volume_scale<'op>(
    input: &VolumeOperator<'op>,
    factor: &ScalarOperator<'op, f32>,
) -> VolumeOperator<'op> {
    VolumeOperator::new(
        OperatorId::new(
            "volume_scale",
            &[*input.metadata.id(), *input.bricks.id(), *factor.id()],
        ),
        |_ctx, _| async { todo!() }.into(),
        |_ctx, _pos| async { todo!() }.into(),
    )
}

fn volume_mean<'op>(input: &VolumeOperator<'op>) -> ScalarOperator<'op, f32> {
    ScalarOperator::new(
        OperatorId::new("volume_scale", &[*input.metadata.id(), *input.bricks.id()]),
        |_ctx, _| async { todo!() }.into(),
    )
}

pub struct RawVolumeSourceState {
    path: PathBuf,
    _file: File,
    mmap: memmap::Mmap,
    metadata: VolumeMetaData,
}

impl<'op> From<&'op RawVolumeSourceState> for VolumeOperator<'op> {
    fn from(value: &'op RawVolumeSourceState) -> Self {
        value.operate()
    }
}

impl RawVolumeSourceState {
    pub fn open(path: PathBuf, metadata: VolumeMetaData) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };

        Ok(Self {
            path,
            _file: file,
            mmap,
            metadata,
        })
    }

    fn operate<'a>(&'a self) -> VolumeOperator<'a> {
        VolumeOperator::new(
            OperatorId::new(
                "RawVolumeSourceState::operate",
                &[Id::from_data(self.path.to_string_lossy().as_bytes()).into()],
            ),
            move |_ctx, _| {
                async move {
                    /*ctx.write_metadata(metadata)*/
                    todo!()
                }
                .into()
            },
            move |ctx: TaskContext, positions| {
                async move {
                    let m = &self.metadata;
                    for pos in positions {
                        let begin = m.brick_begin(pos);
                        if !(begin.0.x < m.dimensions.0.x
                            && begin.0.y < m.dimensions.0.y
                            && begin.0.z < m.dimensions.0.z)
                        {
                            return Err("Brick position is outside of volume".into());
                        }
                        let brick_dim = m.brick_dim(pos).0;
                        let num_voxels = crate::data::hmul(m.brick_size.0) as usize;

                        let voxel_size = std::mem::size_of::<f32>();

                        // Safety: We are zeroing all brick data in a first step.
                        // TODO: We might want to lift this restriction in the future
                        let mut brick_handle: WriteHandleUninit<[MaybeUninit<f32>]> = todo!(); //ctx.alloc_brick(pos, num_voxels)?;
                        let brick_data = &mut *brick_handle;
                        ctx.submit(ctx.spawn_job(move || {
                            brick_data.iter_mut().for_each(|v| {
                                v.write(0.0);
                            });

                            for z in 0..brick_dim.z {
                                for y in 0..brick_dim.y {
                                    let bu8 = voxel_size
                                        * to_linear(
                                            begin.0 + cgmath::vec3(0, y, z),
                                            m.dimensions.0,
                                        );
                                    let eu8 = voxel_size
                                        * to_linear(
                                            begin.0 + cgmath::vec3(brick_dim.x, y, z),
                                            m.dimensions.0,
                                        );

                                    let bf32 = to_linear(cgmath::vec3(0, y, z), m.brick_size.0);
                                    let ef32 =
                                        to_linear(cgmath::vec3(brick_dim.x, y, z), m.brick_size.0);

                                    let in_ = &self.mmap[bu8..eu8];
                                    let out = &mut brick_data[bf32..ef32];
                                    let in_slice: &[f32] = bytemuck::cast_slice(in_);
                                    for (in_, out) in in_slice.iter().zip(out.iter_mut()) {
                                        out.write(*in_);
                                    }
                                }
                            }
                        }))
                        .await;

                        // At this point the thread pool job above has finished and has initialized all bytes
                        // in the brick.
                        unsafe { brick_handle.initialized() };
                    }
                    Ok(())
                }
                .into()
            },
        )
    }
}

fn toy_network(vol: &RawVolumeSourceState, factor: &f32) {
    let volume = vol.into();
    let factor = factor.into();

    let scaled = volume_scale(&volume, &factor);
    let mean = volume_mean(&scaled);
}

fn toy() {
    let metadata = VolumeMetaData {
        dimensions: VoxelPosition((40, 40, 40).into()),
        brick_size: VoxelPosition((32, 32, 32).into()),
    };
    let vol_state = RawVolumeSourceState::open(PathBuf::from("some_path"), metadata).unwrap();

    let factor = 25.0;

    toy_network(&vol_state, &factor);
}
