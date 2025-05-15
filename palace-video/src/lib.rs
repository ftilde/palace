use std::{path::PathBuf, rc::Rc, str::FromStr, sync::Mutex};

use futures::StreamExt;
use id::Identify;
use palace_core::{
    array::TensorMetaData,
    dim::D3,
    dtypes::StaticElementType,
    operator::{DataDescriptor, DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    vec::Vector,
    Error,
};
use video_rs::{
    decode::DecoderSplit,
    ffmpeg::ffi::{av_image_copy_to_buffer, av_seek_frame, AVPixelFormat, AVSEEK_FLAG_BACKWARD},
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

type TensorElement = Vector<D3, u8>;

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

                    let num_elements = metadata.num_chunk_elements();
                    let layout = std::alloc::Layout::array::<TensorElement>(num_elements).unwrap();

                    for pos in positions {
                        let frames = ctx
                            .submit(ctx.spawn_compute(move || {
                                let chunk_info = metadata.chunk_info(pos);
                                let frame_num = chunk_info.begin[0].raw;
                                let time_pts = frame_num as i64 * pts_per_frame;

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

                                let mut frames = Vec::new();
                                loop {
                                    let (t, frame) = loop {
                                        let packet = decoder.reader.read(stream_index)?;
                                        if let Some(frame) = decoder.decoder.decode_raw(packet)? {
                                            let t = frame.packet().dts;

                                            break (t, frame);
                                        }
                                    };
                                    let t = t - start_time;

                                    let frame_num = (t / pts_per_frame) as u32;
                                    let chunk_id = metadata.chunk_index(
                                        &Vector::<D3, _>::new([frame_num, 0, 0]).chunk(),
                                    );
                                    frames.push((chunk_id, frame));

                                    //println!("requested: {}, actual: {}", time_pts, t);
                                    if t >= time_pts {
                                        return Result::<_, video_rs::Error>::Ok(frames);
                                    }
                                }
                            }))
                            .await?;

                        let allocations = frames.iter().map(|(chunk_id, _)| {
                            let data_id =
                                DataDescriptor::new(ctx.current_op_desc().unwrap(), *chunk_id);
                            ctx.alloc_raw(data_id, layout)
                                .map(|v| v.map(|v| v.into_thread_handle()))
                        });

                        let out_chunks = ctx.submit(ctx.group(allocations)).await;
                        let copies = out_chunks.into_iter().zip(frames.into_iter()).filter_map(
                            |(out_chunk_handle, (chunk_id, mut frame))| {
                                if let Ok(mut chunk_handle) = out_chunk_handle {
                                    let chunk_info = metadata.chunk_info(chunk_id);
                                    Some(ctx.spawn_compute(move || {
                                        let chunk_data = &mut *chunk_handle;
                                        assert!(chunk_info.is_full());

                                        let w = chunk_info.logical_dimensions[2].raw as usize;
                                        let h = chunk_info.logical_dimensions[1].raw as usize;

                                        let frame_ptr = unsafe { frame.as_mut_ptr() };
                                        //let frame_format = video_rs::ffmpeg::ffi::AV_PIX_FMT_BGR32;
                                        let source_format = unsafe { (*frame_ptr).format };
                                        let assumed_format = AVPixelFormat::AV_PIX_FMT_RGB24;
                                        assert_eq!(assumed_format as i32, source_format);

                                        let chunk_data =
                                            palace_core::data::fill_uninit(chunk_data, 0);

                                        let bytes_copied = unsafe {
                                            av_image_copy_to_buffer(
                                                chunk_data.as_mut_ptr(),
                                                layout.size() as _,
                                                (*frame_ptr).data.as_ptr() as *const *const u8,
                                                (*frame_ptr).linesize.as_ptr(),
                                                assumed_format,
                                                w as _,
                                                h as _,
                                                layout.align() as _,
                                            )
                                        };

                                        if bytes_copied < 0 {
                                            return Err(video_rs::Error::BackendError(
                                                video_rs::ffmpeg::Error::from(bytes_copied),
                                            ));
                                        }

                                        assert_eq!(bytes_copied, layout.size() as i32);

                                        Result::<_, video_rs::Error>::Ok(chunk_handle)
                                    }))
                                } else {
                                    //println!("{:?} already present, wow!", chunk_id);
                                    None
                                }
                            },
                        );

                        let stream = ctx.submit_unordered(copies);

                        futures::pin_mut!(stream);
                        while let Some(handle) = stream.next().await {
                            let handle = handle?;
                            let handle = handle.into_main_handle(ctx.storage());
                            unsafe { handle.initialized(*ctx) };
                        }
                    }
                    Ok(())
                }
                .into()
            },
        )
        .into()
    }
}
