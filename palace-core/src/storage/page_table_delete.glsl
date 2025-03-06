#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#define ChunkValue float

#include<page_table.glsl>
#include <util.glsl>
//#include <sample.glsl>

layout(scalar, push_constant) uniform PushConsts {
#ifdef MODE_INNER
    PageTablePage to_delete;
#endif
#ifdef MODE_LEAF
    uint64_t chunk_index;
#endif
    PageTablePage root;
} consts;

void main() {
    //uint current_linear = global_position_linear;
    uint current_linear = gl_GlobalInvocationID.x;

    if(current_linear >= 1) {
        return;
    }

#ifdef MODE_INNER
    uint64_t page_table_index = consts.to_delete.page_table_index;
    if(page_table_index == PAGE_TABLE_INDEX_UNUSED) {
        return;
    }
#endif
#ifdef MODE_LEAF
    uint64_t page_table_index = consts.chunk_index;
#endif
    uint level = level(page_table_index);
    uint remove_level = level + 1;
    uvec3 level_indices = page_table_index_to_level_indices(page_table_index);

    PageTablePage node = consts.root;
    for(uint l = PAGE_TABLE_LEVELS; true ; l -= 1) {
        uint level_index = level_indices[PAGE_TABLE_LEVELS-l];
        if(l == remove_level) {
            node.values[level_index] = 0;
            break;
        } else {
            node = PageTablePage(node.values[level_index]);
            if(uint64_t(node) == 0) {
                // Someone has removed a parent node before. We are done here.
                break;
            }
        }
    }
}
