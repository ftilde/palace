use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{ImageMetaData, TensorMetaData, VolumeMetaData},
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

use super::volume::VolumeOperator;

pub struct Splitter {
    size: PixelPosition,
    split_pos: f32,
}

const NUM_CHANNELS: u32 = 4;

impl Splitter {
    pub fn split_events(&mut self, e: &mut EventStream) -> (EventStream, EventStream) {
        let size_r = self.size_r().x().raw as i32;
        let t = |p| p - Vector::<2, i32>::from([0i32, size_r]);

        let mut left = EventStream::with_state(e.latest_state().clone());
        let mut right = EventStream::with_state(e.latest_state().clone().transform(t));

        //TODO: This needs some better handling of (for example) CursorLeft/CursorEntered
        //events
        e.act(|e| {
            if let Some(mouse_state) = &e.state.mouse_state {
                let in_left = mouse_state.pos.x() < size_r;
                if in_left {
                    left.add(e);
                } else {
                    right.add(e.transform(t));
                };
                EventChain::Consumed
            } else {
                e.into()
            }
        });
        (left, right)
    }

    pub fn new(size: PixelPosition, split_pos: f32) -> Self {
        assert!(0.0 <= split_pos && split_pos <= 1.0);
        Splitter { size, split_pos }
    }
    pub fn size_l(&self) -> PixelPosition {
        self.size.map_element(1, |v| {
            ((v.raw as f32 * self.split_pos).round() as u32).into()
        })
    }
    pub fn metadata_l(&self) -> ImageMetaData {
        let s = self.size_l();
        //TODO: Allow smaller chunk sizes
        TensorMetaData {
            dimensions: s,
            chunk_size: s.local(),
        }
    }
    pub fn size_r(&self) -> PixelPosition {
        let mut r = self.size;
        r[1] = r[1] - self.size_l()[1];
        r
    }
    pub fn metadata_r(&self) -> ImageMetaData {
        let s = self.size_r();
        TensorMetaData {
            dimensions: s,
            chunk_size: s.local(),
        }
    }
    pub fn metadata_out(&self) -> VolumeMetaData {
        let s = self.size.add_dim(2, NUM_CHANNELS.into());
        VolumeMetaData {
            dimensions: s,
            chunk_size: s.local(),
        }
    }

    pub fn render(self, input_l: VolumeOperator, input_r: VolumeOperator) -> VolumeOperator {
        #[derive(Copy, Clone, AsStd140, GlslStruct)]
        struct PushConstants {
            dim_out: cgmath::Vector2<u32>,
            dim_l: cgmath::Vector2<u32>,
            dim_r: cgmath::Vector2<u32>,
        }
        const SHADER: &'static str = r#"
#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) readonly buffer InputBufferL{
    float values[];
} input_l;

layout(std430, binding = 1) readonly buffer InputBufferR{
    float values[];
} input_r;

layout(std430, binding = 2) buffer OutputBuffer {
    float values[];
} output_buf;

declare_push_consts(consts);

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint g_id_out = out_pos.x + out_pos.y * consts.dim_out.x;

    if(all(lessThan(out_pos, consts.dim_out))) {
        vec4 out_val = vec4(0.0);

        uvec2 pos_l = out_pos;
        if(pos_l.x < consts.dim_l.x && pos_l.y < consts.dim_l.y) {
            uint g_id_l = pos_l.x + pos_l.y * consts.dim_l.x;
            for(int c=0; c<4; ++c) {
                out_val[c] = input_l.values[g_id_l*4 + c];
            }
        }

        uvec2 pos_r = out_pos - uvec2(consts.dim_l.x, 0);
        if(all(lessThan(pos_r, consts.dim_r))) {
            uint g_id_r = pos_r.x + pos_r.y * consts.dim_r.x;
            for(int c=0; c<4; ++c) {
                out_val[c] = input_r.values[g_id_r*4 + c];
            }
        }

        for(int c=0; c<4; ++c) {
            output_buf.values[g_id_out*4 + c] = out_val[c];
        }
    }
}
"#;

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

                    assert_eq!(m_l.dimensions.drop_dim(2), this.metadata_l().dimensions);
                    assert_eq!(m_r.dimensions.drop_dim(2), this.metadata_r().dimensions);

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
                        let global_size = chunk_size.drop_dim(2).add_dim(0, 1);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(PushConstants {
                                dim_out: m.dimensions.drop_dim(2).into_elem::<u32>().into(),
                                dim_l: m_l.dimensions.drop_dim(2).into_elem::<u32>().into(),
                                dim_r: m_r.dimensions.drop_dim(2).into_elem::<u32>().into(),
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
