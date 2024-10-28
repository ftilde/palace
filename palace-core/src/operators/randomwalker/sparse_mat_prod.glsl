#version 450

#extension GL_EXT_scalar_block_layout : require

#include <randomwalker_shared.glsl>

layout (local_size_x = 1024) in;

layout(std430, binding = 0) readonly buffer MatValues {
    float values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_values;

layout(std430, binding = 1) readonly buffer MatIndex {
    uint values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_index;

layout(std430, binding = 2) buffer X {
    float values[NUM_ROWS];
} x;

layout(std430, binding = 3) buffer Vec {
    float values[NUM_ROWS];
} result;

//declare_push_consts(consts);

void main() {
    uint row = gl_GlobalInvocationID.x;

    if(row >= NUM_ROWS) {
        return;
    }

    float sum = 0.0;
    for(int i=0; i<MAX_ENTRIES_PER_ROW; ++i) {
        uint col = mat_index.values[row][i];
        if(col != MAT_INDEX_EMPTY) {
            sum += mat_values.values[row][i] * x.values[col];
        }
    }
    result.values[row] = sum;
}
