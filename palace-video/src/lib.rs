use std::{
    path::PathBuf,
    rc::Rc,
    str::FromStr,
    sync::{Arc, Mutex},
};

use futures::StreamExt;
use id::Identify;
use palace_core::{
    array::TensorMetaData,
    dim::D3,
    dtypes::StaticElementType,
    operator::{DataParam, OperatorDescriptor},
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

impl VideoSourceStateInner {
    fn pts_to_frame_number(&self, pts: i64) -> u32 {
        (pts / self.pts_per_frame) as u32
    }
    fn frame_number_to_pts(&self, frame_num: u32) -> i64 {
        frame_num as i64 * self.pts_per_frame
    }
}
impl VideoSourceState {
    pub fn open(loc: Location) -> Result<Self, Error> {
        let loc_path = loc.as_path().to_owned();
        let decoder = Decoder::new(loc)?;

        let mut duration = decoder.duration()?;
        let fps = decoder.frame_rate() as f64;
        let frame_size = decoder.size_out();
        let mut n_frames = decoder.frames()?;

        let (decoder, reader, reader_stream_index) = decoder.into_parts();

        if duration.into_value().is_none_or(|v| v < 0) {
            let duration_global_pts = unsafe { *reader.input.as_ptr() }.duration;
            duration = video_rs::Time::new(
                Some(duration_global_pts),
                video_rs::ffmpeg::ffi::AV_TIME_BASE_Q.into(),
            );
        }

        if n_frames == 0 {
            n_frames = (duration.as_secs_f64() * fps) as u64;
        }
        let duration_pts = duration
            .with_time_base(decoder.time_base())
            .into_value()
            .unwrap();
        let pts_per_frame = duration_pts / n_frames as i64;
        //println!(
        //    "Duration (pts) {} | n_frames {} | pts per frame {}",
        //    duration_pts, n_frames, pts_per_frame
        //);

        let metadata = TensorMetaData {
            dimensions: Vector::<D3, _>::new([n_frames as u32, frame_size.1, frame_size.0])
                .global(),
            chunk_size: Vector::<D3, _>::new([1u32, frame_size.1, frame_size.0]).local(),
        };

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
                    let inner = &*this.inner;

                    let num_elements = metadata.num_chunk_elements();
                    let layout = std::alloc::Layout::array::<TensorElement>(num_elements).unwrap();

                    for pos in positions {
                        if ctx.storage().exists_in_storage(
                            ctx.data_descriptor(pos).unwrap().id,
                            palace_core::storage::DataVersion::Final,
                        ) {
                            //println!("Welp, {:?} already there", pos);

                            let data_id = ctx.data_descriptor(pos).unwrap().id;
                            if ctx
                                .storage()
                                .is_readable(data_id, palace_core::storage::DataVersion::Final)
                            {
                                ctx.storage().add_new_data_hint(data_id);
                            }

                            //This doesn't work because (likely) because the data may not be initialized yet.

                            //is_currently_being fulfilled in runtime.rs (or similar) is very flawed for this model.
                            //problem:
                            //    - data is requested
                            //    - task for that data started
                            //    - another task ist produces the data
                            //    - it does not get transferred because the original task "currently fulfills" it
                            //    - original task does not fulfill it

                            //With the fake fulfillments (add_new_data_hint) we get:
                            //    - data is requested
                            //    - task for that data is started
                            //    - another task produces wants to produce the data (allocates the slot)
                            //    - task 1 notices that and (unconditionally) hints fulfillment
                            //    - runtime tries to make data available, but it isn't yet (other task is not done with completion)
                            //
                            //Solution (???):
                            //    - Only emit hint when data is actually readable (???)

                            continue;
                        }

                        let frames = ctx
                            .submit(ctx.spawn_compute(move || {
                                let chunk_info = metadata.chunk_info(pos);
                                let requested_frame_num = chunk_info.begin[0].raw;
                                let time_pts = inner.frame_number_to_pts(requested_frame_num);

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
                                'outer: loop {
                                    let (t, frame) = loop {
                                        let packet = match decoder.reader.read(stream_index) {
                                            Ok(p) => p,
                                            Err(video_rs::Error::ReadExhausted) => break 'outer,
                                            Err(e) => return Err(e),
                                        };
                                        if let Some(frame) = decoder.decoder.decode_raw(packet)? {
                                            let t = frame.packet().dts;

                                            break (t, frame);
                                        }
                                    };
                                    let t = t - start_time;

                                    let is_key = frame.is_key();
                                    frames.push((t, Arc::new(frame)));

                                    //println!("push {} ", t);

                                    if t >= time_pts && is_key {
                                        break;
                                    }
                                }

                                let min_frame =
                                    inner.pts_to_frame_number(frames.first().unwrap().0);
                                let max_frame = inner.pts_to_frame_number(frames.last().unwrap().0);

                                let mut out_frames = Vec::with_capacity(frames.len());

                                assert!(
                                    min_frame <= pos.raw() as u32, // && pos.raw() as u32 <= max_frame,
                                    "Requested frame not included"
                                );

                                for frame_num in min_frame..=max_frame.max(requested_frame_num) {
                                    let pts = inner.frame_number_to_pts(frame_num);
                                    let right_i =
                                        frames.partition_point(|v| v.0 < pts).min(frames.len() - 1);
                                    let left_i = right_i.saturating_sub(1);
                                    let right = &frames[right_i];
                                    let left = &frames[left_i];

                                    let chunk_id = metadata.chunk_index(
                                        &Vector::<D3, _>::new([frame_num, 0, 0]).chunk(),
                                    );

                                    if (pts - left.0).abs() < (right.0 - pts).abs() {
                                        out_frames.push((chunk_id, left.1.clone()));
                                    } else {
                                        out_frames.push((chunk_id, right.1.clone()));
                                    };
                                    //println!("push {} for {}, chunk {:?}", r, pts, chunk_id);
                                }
                                return Result::<_, video_rs::Error>::Ok(out_frames);
                            }))
                            .await?;

                        let allocations = frames.iter().map(|(chunk_id, _)| {
                            let data_id = ctx.data_descriptor(*chunk_id).unwrap();
                            ctx.alloc_raw(data_id, layout)
                                .map(|v| v.map(|v| v.into_thread_handle()))
                        });

                        let out_chunks = ctx.submit(ctx.group(allocations)).await;
                        let copies = out_chunks.into_iter().zip(frames.into_iter()).filter_map(
                            |(out_chunk_handle, (chunk_id, frame))| {
                                if let Ok(mut chunk_handle) = out_chunk_handle {
                                    let chunk_info = metadata.chunk_info(chunk_id);
                                    Some(ctx.spawn_compute(move || {
                                        let chunk_data = &mut *chunk_handle;
                                        assert!(chunk_info.is_full());

                                        let w = chunk_info.logical_dimensions[2].raw as usize;
                                        let h = chunk_info.logical_dimensions[1].raw as usize;

                                        let frame_ptr = unsafe { frame.as_ptr() };
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
                                    let data_id = ctx.data_descriptor(chunk_id).unwrap().id;
                                    if ctx.storage().is_readable(
                                        data_id,
                                        palace_core::storage::DataVersion::Final,
                                    ) {
                                        ctx.storage().add_new_data_hint(data_id);
                                    }
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
