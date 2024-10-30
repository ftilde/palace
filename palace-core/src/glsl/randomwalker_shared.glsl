#define MAT_INDEX_EMPTY 0xffffffff
#define TENSOR_TO_VEC_TABLE_SEED 0xffffffff
#define UNSEEDED -2.0

bool is_seed_value(float val) {
    return val != UNSEEDED;
}

#define MAT_PROD_ROW(mat_index, mat_values, x, result) { \
    result = 0.0; \
    for(int i=0; i<MAX_ENTRIES_PER_ROW; ++i) { \
        uint col = (mat_index)[row][i]; \
        if(col != MAT_INDEX_EMPTY) { \
            result += (mat_values)[row][i] * (x)[col]; \
        } \
    } \
}
