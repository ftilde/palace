use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{ImageMetaData, TensorMetaData},
    data::{PixelPosition, Vector},
    event::{EventChain, EventStream},
    operator::OperatorId,
    operators::tensor::TensorOperator,
    vulkan::{
        pipeline::{ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::tensor::FrameOperator;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
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

#[derive(Clone)]
pub struct Splitter {
    size: PixelPosition,
    split_pos: f32,
    split_dim: usize,
}

impl Splitter {
    pub fn split_events(&mut self, e: &mut EventStream) -> (EventStream, EventStream) {
        let size_last = self.size_last()[self.split_dim].raw as i32;
        let t = |p| p - Vector::<2, i32>::fill(0).map_element(self.split_dim, |_| size_last);

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
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_scalar_block_layout : require

layout (local_size_x = 32, local_size_y = 32) in;

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
        // Shaders assume x = dim 0 and y = dim 1
        let split_dim = 1 - self.split_dim as u32;

        TensorOperator::with_state(
            OperatorId::new("splitter")
                .dependent_on(&input_l)
                .dependent_on(&input_r),
            self.metadata_out(),
            (input_l.clone(), input_r.clone(), self),
            move |ctx, m_out| async move { ctx.write(*m_out) }.into(),
            move |ctx, positions, (input_l, input_r, this)| {
                async move {
                    let device = ctx.vulkan_device();

                    let access_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ,
                    };

                    let (m_l, img_l, m_r, img_r) = futures::join! {
                        ctx.submit(input_l.metadata.request_scalar()),
                        ctx.submit(input_l.chunks.request_gpu(device.id, Vector::fill(0.into()), access_info)),
                        ctx.submit(input_r.metadata.request_scalar()),
                        ctx.submit(input_r.chunks.request_gpu(device.id, Vector::fill(0.into()), access_info)),
                    };

                    assert_eq!(m_l.dimensions, this.metadata_first().dimensions);
                    assert_eq!(m_r.dimensions, this.metadata_last().dimensions);

                    let m = this.metadata_out();

                    let pipeline = device
                        .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                            ComputePipeline::new(device, (SHADER, ShaderDefines::new().push_const_block::<PushConstants>()), true)
                        });

                    assert!(positions.len() == 1);
                    let pos = *positions.first().unwrap();
                    assert_eq!(pos, Vector::fill(0.into()));

                    let out_info = m.chunk_info(pos);
                    let gpu_brick_out = ctx
                        .alloc_slot_gpu(device, pos, out_info.mem_elements())
                        .unwrap();

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &img_l,
                            &img_r,
                            &gpu_brick_out,
                        ]);

                        let chunk_size = m.chunk_size.raw();
                        let global_size = chunk_size.add_dim(0, 1);

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
