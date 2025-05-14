use std::{path::PathBuf, rc::Rc, str::FromStr, sync::Mutex};

use futures::StreamExt;
use id::Identify;
use palace_core::{
    array::TensorMetaData,
    dim::{D3, D4},
    dtypes::StaticElementType,
    operator::{DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    task::RequestStream,
    vec::Vector,
    Error,
};
use video_rs::{Decoder, Location};

#[derive(Clone)]
pub struct VideoSourceState {
    inner: Rc<VideoSourceStateInner>,
}

impl Identify for VideoSourceState {
    fn id(&self) -> id::Id {
        self.inner.loc.id()
    }
}

pub struct VideoSourceStateInner {
    metadata: TensorMetaData<D3>,
    //embedding_data: TensorEmbeddingData<D3>,
    decoder: Mutex<Decoder>,
    loc: PathBuf,
}

type TensorElement = Vector<D4, u8>;

pub fn open(loc: &str) -> Result<TensorOperator<D3, StaticElementType<TensorElement>>, Error> {
    let loc = if loc.starts_with("http") {
        Location::Network(video_rs::Url::from_str(loc)?)
    } else {
        Location::File(PathBuf::from_str(loc)?)
    };
    Ok(VideoSourceState::open(loc)?.operate())
}

impl VideoSourceState {
    pub fn open(loc: Location) -> Result<Self, Error> {
        let loc_path = loc.as_path().to_owned();
        let decoder = Decoder::new(loc)?;
        let n_frames = decoder.frames()?;
        let frame_size = decoder.size_out();

        let metadata = TensorMetaData {
            dimensions: Vector::<D3, _>::new([n_frames as u32, frame_size.1, frame_size.0])
                .global(),
            chunk_size: Vector::<D3, _>::new([1u32, frame_size.1, frame_size.0]).local(),
        };

        let decoder = Mutex::new(decoder);
        let inner = VideoSourceStateInner {
            metadata,
            decoder,
            loc: loc_path,
        };

        Ok(VideoSourceState {
            inner: Rc::new(inner),
        })
    }

    pub fn operate(&self) -> TensorOperator<D3, StaticElementType<TensorElement>> {
        TensorOperator::with_state(
            OperatorDescriptor::with_name("VideoSourceState::operate"),
            Default::default(),
            self.inner.metadata.clone(),
            DataParam(self.clone()),
            move |ctx, positions, _location, this| {
                //println!("Positions: {}", positions.len());
                async move {
                    let metadata = &this.inner.metadata;
                    let decoder = &this.inner.decoder;

                    let allocations = positions
                        .into_iter()
                        .map(|chunk_id| (ctx.alloc_slot(chunk_id, &metadata.chunk_size), chunk_id));
                    let stream = ctx.submit_unordered_with_data(allocations).then_req(
                        *ctx,
                        |(chunk_handle, chunk_id)| {
                            let mut chunk_handle = chunk_handle.into_thread_handle();
                            ctx.spawn_compute(move || {
                                let chunk_info = metadata.chunk_info(chunk_id);
                                let frame_num = chunk_info.begin[0].raw;
                                let (_t, in_chunk) = {
                                    let mut decoder = decoder.lock().unwrap();

                                    decoder.seek(frame_num as i64)?;
                                    decoder.decode()?
                                };

                                let chunk_data = &mut *chunk_handle;
                                assert!(chunk_info.is_full());

                                let w = chunk_info.logical_dimensions[2];
                                let h = chunk_info.logical_dimensions[1];

                                let mut out_chunk =
                                    palace_core::data::chunk_mut(chunk_data, &chunk_info);

                                for y in 0..h.raw as usize {
                                    for x in 0..w.raw as usize {
                                        let r = *in_chunk.get((y, x, 0)).unwrap();
                                        let g = *in_chunk.get((y, x, 1)).unwrap();
                                        let b = *in_chunk.get((y, x, 2)).unwrap();
                                        let val = Vector::new([r, g, b, 255]);

                                        out_chunk.get_mut((0, y, x)).unwrap().write(val);
                                    }
                                }

                                Result::<_, video_rs::Error>::Ok(chunk_handle)
                            })
                        },
                    );

                    futures::pin_mut!(stream);
                    while let Some(handle) = stream.next().await {
                        let handle = handle?;
                        let handle = handle.into_main_handle(ctx.storage());
                        unsafe { handle.initialized(*ctx) };
                    }
                    Ok(())
                }
                .into()
            },
        )
        .into()
    }
}
