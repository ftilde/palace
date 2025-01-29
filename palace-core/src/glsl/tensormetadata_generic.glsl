#include<vec.glsl>

struct TensorMetaData(_N) {
    uint[_N] dimensions;
    uint[_N] chunk_size;
};

uint[_N] dim_in_bricks(TensorMetaData(_N) vm) {
    return div_round_up(vm.dimensions, vm.chunk_size);
}

uint[_N] chunk_pos(TensorMetaData(_N) vm, uint[_N] global_pos) {
    return div(global_pos, (vm).chunk_size);
}

uint[_N] chunk_begin(TensorMetaData(_N) vm, uint[_N] chunk_pos) {
    return mul(chunk_pos, (vm).chunk_size);
}
