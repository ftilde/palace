#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_int64: require
//#extension GL_EXT_debug_printf : enable

#include <page_table.glsl>
#include <util.glsl>
#include <size_util.glsl>

layout(scalar, push_constant) uniform PushConsts {
    PageTablePage root;
    uint num_elements;
    uint _padding;
} consts;

struct PosAndChunk {
    uint64_t chunk_id;
    uint64_t buffer_addr;
};

layout(std430, binding = 0) readonly buffer PosAndChunkBuffer {
    PosAndChunk values[];
} pos_and_chunk;

layout(std430, binding = 1) buffer PageTablePool {
    PageTablePage values[];
} page_table_pool;

layout(std430, binding = 2) buffer PageTablePoolPos {
    uint next;
} page_table_pool_pos;

PageTablePage pop_page() {
    //TODO add check here if we reduce page_table_pool_size below 2*num_elements
    uint pos = atomicAdd(page_table_pool_pos.next, 1);
    return page_table_pool.values[pos];
}

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= consts.num_elements) {
        return;
    }

    PosAndChunk pac = pos_and_chunk.values[current_linear];

    uvec3 level_indices = page_table_index_to_level_indices(pac.chunk_id);
    uint index0 = level_indices[0];
    uint index1 = level_indices[1];
    uint index2 = level_indices[2];

    //debugPrintfEXT("insert %lu: %d %d %d \n", pac.chunk_id, index0, index1, index2);

    while(consts.root.values[index0] == 0) {
        PageTablePage page_buf = pop_page();

        //debugPrintfEXT("set page level1 %d %lu \n", index0, uint64_t(page_buf));

        uint64_t prev = atomicCompSwap(consts.root.values[index0], 0L, uint64_t(page_buf));
        bool buf_used = prev == 0L;
        if(buf_used) {
            page_buf.page_table_index = with_level(pac.chunk_id, 2);
        } else {
            //TODO: Can we put it back? hmm... same below
            page_buf.page_table_index = PAGE_TABLE_INDEX_UNUSED;
        }
    }
    PageTablePage l1 = PageTablePage(consts.root.values[index0]);

    while(l1.values[index1] == 0) {
        PageTablePage page_buf = pop_page();

        //debugPrintfEXT("set page level2 %d %lu \n", index1, uint64_t(page_buf));

        uint64_t prev = atomicCompSwap(l1.values[index1], 0L, uint64_t(page_buf));
        bool buf_used = prev == 0L;
        if(buf_used) {
            page_buf.page_table_index = with_level(pac.chunk_id, 1);
        } else {
            page_buf.page_table_index = PAGE_TABLE_INDEX_UNUSED;
        }
    }
    PageTablePage l2 = PageTablePage(l1.values[index1]);

    l2.values[index2] = pac.buffer_addr;
}
