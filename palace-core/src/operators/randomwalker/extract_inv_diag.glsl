#version 450

#extension GL_EXT_scalar_block_layout : require

layout(std430, binding = 0) readonly buffer MatValues {
    float values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_values;

layout(std430, binding = 1) readonly buffer MatIndex {
    uint values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_index;

layout(std430, binding = 2) buffer Vec {
    float values[NUM_ROWS];
} result;

//declare_push_consts(consts)

void main() {
    uint row = gl_GlobalInvocationID.x;

    if(row >= NUM_ROWS) {
        return;
    }

    float clip_value = 1e-6;

    for(int i=0; i<MAX_ENTRIES_PER_ROW; ++i) {
        if(mat_index.values[row][i] == row) {
            float diag_value = mat_values.values[row][i];
            result.values[row] = 1.0/max(clip_value, diag_value);
        }
    }
}
