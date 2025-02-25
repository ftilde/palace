use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use id::Identify;

use crate::{
    array::{ChunkIndex, ImageMetaData, TensorMetaData},
    data::{PixelPosition, Vector},
    dim::*,
    event::{EventChain, EventStream},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, LocalSizeConfig},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::tensor::FrameOperator;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[derive(Clone, Identify, PartialEq, Eq)]
pub enum SplitDirection {
    Horizontal,
    Vertical,
}

impl SplitDirection {
    fn dim(&self) -> usize {
        match self {
            SplitDirection::Horizontal => 1,
            SplitDirection::Vertical => 0,
        }
    }
}

#[derive(Clone, Identify)]
pub struct Splitter {
    size: PixelPosition,
    split_pos: f32,
    split_dim: usize,
}

impl Splitter {
    pub fn split_events(&mut self, e: &mut EventStream) -> (EventStream, EventStream) {
        let size_last = self.size_last()[self.split_dim].raw as i32;
        let t = |p| p - Vector::<D2, i32>::fill(0).map_element(self.split_dim, |_| size_last);

        let mut first = EventStream::with_state(e.latest_state().clone());
        let mut last = EventStream::with_state(e.latest_state().clone().transform(t));

        //TODO: This needs some better handling of (for example) CursorLeft/CursorEntered
        //events
        e.act(|e| {
            if let Some(mouse_state) = &e.state.mouse_state {
                let in_first = mouse_state.pos[self.split_dim] < size_last;
                if in_first {
                    first.add(e);
                } else {
                    last.add(e.transform(t));
                };
                EventChain::Consumed
            } else {
                e.into()
            }
        });
        (first, last)
    }

    pub fn new(size: PixelPosition, split_pos: f32, split_dir: SplitDirection) -> Self {
        assert!(0.0 <= split_pos && split_pos <= 1.0);
        Splitter {
            size,
            split_pos,
            split_dim: split_dir.dim(),
        }
    }
    pub fn size_first(&self) -> PixelPosition {
        self.size.map_element(self.split_dim, |v| {
            ((v.raw as f32 * self.split_pos).round() as u32).into()
        })
    }
    pub fn metadata_first(&self) -> ImageMetaData {
        let s = self.size_first();
        //TODO: Allow smaller chunk sizes
        TensorMetaData {
            dimensions: s,
            chunk_size: s.local(),
        }
    }
    pub fn size_last(&self) -> PixelPosition {
        let mut r = self.size;
        r[self.split_dim] = r[self.split_dim] - self.size_first()[self.split_dim];
        r
    }
    pub fn metadata_last(&self) -> ImageMetaData {
        let s = self.size_last();
        TensorMetaData {
            dimensions: s,
            chunk_size: s.local(),
        }
    }
    pub fn metadata_out(&self) -> ImageMetaData {
        TensorMetaData {
            dimensions: self.size,
            chunk_size: self.size.local(),
        }
    }

    pub fn render(self, input_l: FrameOperator, input_r: FrameOperator) -> FrameOperator {
        #[derive(Copy, Clone, AsStd140, GlslStruct)]
        struct PushConstants {
            size_out: cgmath::Vector2<u32>,
            size_first: cgmath::Vector2<u32>,
            size_last: cgmath::Vector2<u32>,
            split_dim: u32, //TODO we could also make this a constant in the shader...
        }
        const SHADER: &'static str = r#"
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_scalar_block_layout : require

layout(scalar, binding = 0) readonly buffer InputBufferL{
    u8vec4 values[];
} input_l;

layout(scalar, binding = 1) readonly buffer InputBufferR{
    u8vec4 values[];
} input_r;

layout(scalar, binding = 2) buffer OutputBuffer {
    u8vec4 values[];
} output_buf;

declare_push_consts(consts);

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint g_id_out = out_pos.x + out_pos.y * consts.size_out.x;

    if(all(lessThan(out_pos, consts.size_out))) {
        u8vec4 out_val;

        if(out_pos[consts.split_dim] < consts.size_first[consts.split_dim]) {
            uvec2 pos_first = out_pos;

            uint g_id_l = pos_first.x + pos_first.y * consts.size_first.x;
            out_val = input_l.values[g_id_l];
        } else {
            uvec2 pos_last = out_pos;
            pos_last[consts.split_dim] -= consts.size_first[consts.split_dim];

            uint g_id_r = pos_last.x + pos_last.y * consts.size_last.x;
            out_val = input_r.values[g_id_r];
        }

        output_buf.values[g_id_out] = out_val;
    }
}
"#;

        TensorOperator::with_state(
            op_descriptor!(),
            Default::default(),
            self.metadata_out(),
            (input_l, input_r, DataParam(self)),
            move |ctx, positions, loc, (input_l, input_r, this)| {
                async move {
                    // Shaders assume x = dim 0 and y = dim 1
                    let split_dim = 1 - this.split_dim as u32;

                    let device = ctx.preferred_device(loc);

                    let access_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ,
                    };

                    let m_l = input_l.metadata;
                    let m_r = input_r.metadata;
                    let (img_l, img_r) = futures::join! {
                        ctx.submit(input_l.chunks.request_gpu(device.id, ChunkIndex(0), access_info)),
                        ctx.submit(input_r.chunks.request_gpu(device.id, ChunkIndex(0), access_info)),
                    };


                    assert_eq!(m_l.dimensions, this.metadata_first().dimensions);
                    assert_eq!(m_r.dimensions, this.metadata_last().dimensions);

                    let m = this.metadata_out();

                    let pipeline = device
                        .request_state((), |device, ()| {
                            ComputePipelineBuilder::new(
                                Shader::new(SHADER)
                                .push_const_block::<PushConstants>()
                            )
                                .local_size(LocalSizeConfig::Auto2D)
                                .use_push_descriptor(true)
                                .build(device)
                        })?;

                    assert!(positions.len() == 1);
                    let pos = *positions.first().unwrap();
                    assert_eq!(pos, ChunkIndex(0));

                    let out_info = m.chunk_info(pos);
                    let gpu_brick_out = ctx.submit(ctx
                        .alloc_slot_gpu(device, pos, &out_info.mem_dimensions)).await;

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &img_l,
                            &img_r,
                            &gpu_brick_out,
                        ]);

                        let chunk_size = m.chunk_size.raw();
                        let global_size = chunk_size.push_dim_large(1);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(PushConstants {
                                size_out: m.dimensions.into_elem::<u32>().into(),
                                size_first: m_l.dimensions.into_elem::<u32>().into(),
                                size_last: m_r.dimensions.into_elem::<u32>().into(),
                                split_dim,
                            });
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch3d(global_size);
                        }
                    });

                    unsafe {
                        gpu_brick_out.initialized(*ctx, SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_WRITE,
                        })
                    };

                    Ok(())
                }
                .into()
            },
        )
    }
}
