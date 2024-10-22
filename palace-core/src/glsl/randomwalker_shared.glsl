#define MAT_INDEX_EMPTY 0xffffffff
#define TENSOR_TO_VEC_TABLE_SEED 0xffffffff
#define UNSEEDED -2.0

bool is_seed_value(float val) {
    return val != UNSEEDED;
}
