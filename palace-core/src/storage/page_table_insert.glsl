#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_int64: require
//#extension GL_EXT_debug_printf : enable

#include<page_table.glsl>
#include <util.glsl>
//#include <sample.glsl>

layout(scalar, push_constant) uniform PushConsts {
    PageTablePage root;
    uint64_t chunk_id;
    uint64_t buffer_addr;
    PageTablePage buf_level_1;
    PageTablePage buf_level_2;
} consts;

void main() {
    //uint current_linear = global_position_linear;
    uint current_linear = gl_GlobalInvocationID.x;

    if(current_linear >= 1) {
        return;
    }

    // Init pt_index of potential pages in case we don't need them
    consts.buf_level_1.page_table_index = PAGE_TABLE_INDEX_UNUSED;
    consts.buf_level_2.page_table_index = PAGE_TABLE_INDEX_UNUSED;

    uvec3 level_indices = page_table_index_to_level_indices(consts.chunk_id);
    uint index0 = level_indices[0];
    uint index1 = level_indices[1];
    uint index2 = level_indices[2];

    //debugPrintfEXT("insert %lu: %d %d %d \n", consts.chunk_id, index0, index1, index2);

    while(consts.root.values[index0] == 0) {
        //debugPrintfEXT("set page level1 %d %lu \n", index0, uint64_t(consts.buf_level_1));

        uint64_t prev = atomicCompSwap(consts.root.values[index0], 0L, uint64_t(consts.buf_level_1));
        bool buf_used = prev == 0L;
        if(buf_used) {
            consts.buf_level_1.page_table_index = with_level(consts.chunk_id, 2);
        }
    }
    PageTablePage l1 = PageTablePage(consts.root.values[index0]);

    while(l1.values[index1] == 0) {
        //debugPrintfEXT("set page level2 %d %lu \n", index1, uint64_t(consts.buf_level_2));

        uint64_t prev = atomicCompSwap(l1.values[index1], 0L, uint64_t(consts.buf_level_2));
        bool buf_used = prev == 0L;
        if(buf_used) {
            consts.buf_level_2.page_table_index = with_level(consts.chunk_id, 1);
        }
    }
    PageTablePage l2 = PageTablePage(l1.values[index1]);

    l2.values[index2] = consts.buffer_addr;
}
