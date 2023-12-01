struct TensorMetaData(_N) {
    uint[_N] dimensions;
    uint[_N] chunk_size;
};

uint[_N] dim_in_bricks(TensorMetaData(_N) vm) {
    return div_round_up(vm.dimensions, vm.chunk_size);
}
