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
use video_rs::{
    decode::DecoderSplit,
    ffmpeg::ffi::{av_seek_frame, AVSEEK_FLAG_BACKWARD},
    Decoder, Location, Reader,
};

#[derive(Clone)]
pub struct VideoSourceState {
    inner: Rc<VideoSourceStateInner>,
}

impl Identify for VideoSourceState {
    fn id(&self) -> id::Id {
        self.inner.loc.id()
    }
}

struct DecoderParts {
    decoder: DecoderSplit,
    reader: Reader,
    reader_stream_index: usize,
}

pub struct VideoSourceStateInner {
    metadata: TensorMetaData<D3>,
    //embedding_data: TensorEmbeddingData<D3>,
    decoder: Mutex<DecoderParts>,
    loc: PathBuf,
    pts_per_frame: i64,
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

        let duration_pts = decoder.duration()?.into_value().unwrap();
        let n_frames = decoder.frames()?;
        let pts_per_frame = duration_pts / n_frames as i64;
        let frame_size = decoder.size_out();

        let metadata = TensorMetaData {
            dimensions: Vector::<D3, _>::new([n_frames as u32, frame_size.1, frame_size.0])
                .global(),
            chunk_size: Vector::<D3, _>::new([1u32, frame_size.1, frame_size.0]).local(),
        };

        let (decoder, reader, reader_stream_index) = decoder.into_parts();
        let decoder = DecoderParts {
            decoder,
            reader,
            reader_stream_index,
        };

        let decoder = Mutex::new(decoder);
        let inner = VideoSourceStateInner {
            metadata,
            decoder,
            loc: loc_path,
            pts_per_frame,
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
                    let pts_per_frame = this.inner.pts_per_frame;

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
                                let time_pts = frame_num as i64 * pts_per_frame;
                                let in_chunk = 'ret: loop {
                                    let mut decoder = decoder.lock().unwrap();
                                    let stream_index = decoder.reader_stream_index;

                                    let stream = decoder.reader.input.stream(stream_index).unwrap();

                                    let start_time = stream.start_time();

                                    let timestamp = time_pts + start_time;

                                    unsafe {
                                        match av_seek_frame(
                                            decoder.reader.input.as_mut_ptr(),
                                            stream_index as _,
                                            timestamp,
                                            AVSEEK_FLAG_BACKWARD,
                                        ) {
                                            s if s >= 0 => Ok(()),
                                            e => Err(video_rs::ffmpeg::Error::from(e)),
                                        }
                                    }
                                    .map_err(video_rs::Error::BackendError)
                                    .inspect(|_| decoder.decoder.reset())?;

                                    loop {
                                        let (t, frame) = loop {
                                            let packet = decoder.reader.read(stream_index)?;
                                            if let Some(p) = decoder.decoder.decode(packet)? {
                                                break p;
                                            }
                                        };
                                        let t = t.into_value().unwrap();
                                        let t = t - start_time;

                                        //println!("requested: {}, actual: {}", time_pts, t);
                                        if t >= time_pts {
                                            break 'ret frame;
                                        }
                                    }
                                };
                                //println!("===============================");

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
