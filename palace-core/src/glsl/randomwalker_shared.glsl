#define MAT_INDEX_EMPTY 0xffffffff
#define TENSOR_TO_VEC_TABLE_SEED 0xffffffff
#define TENSOR_TO_VEC_TABLE_EMPTY 0xfffffffe
#define UNSEEDED -2.0

bool is_seed_value(float val) {
    return val != UNSEEDED;
}

#define MAT_PROD_ROW(mat_index, mat_values, x, row, result) { \
    result = 0.0; \
    for(int i=0; i<MAX_ENTRIES_PER_ROW; ++i) { \
        uint col = (mat_index)[row][i]; \
        if(col != MAT_INDEX_EMPTY) { \
            result += (mat_values)[row][i] * (x)[col]; \
        } \
    } \
}

#define DOT_PRODUCT_INIT(x, y, row, num_rows, local_result, global_result) { \
    float val;\
    if(row < num_rows) {\
        val = x[row] * y[row];\
    } else {\
        val = 0.0;\
    }\
\
    float sg_agg = subgroupAdd(val);\
\
    if(gl_SubgroupInvocationID == 0) {\
        local_result[gl_SubgroupID] = sg_agg;\
    }\
\
    uint s = 1;\
\
    while(s < gl_NumSubgroups) {\
        barrier();\
        if(gl_SubgroupInvocationID == 0) {\
            if(((gl_SubgroupID & ((2*s)-1)) == 0) && (gl_SubgroupID + s < gl_NumSubgroups)) {\
                local_result[gl_SubgroupID] += local_result[gl_SubgroupID+s];\
            }\
        }\
        s *= 2;\
    }\
\
    barrier();\
\
    if(local_index_subgroup_order == 0) {\
        global_result[workgroup_id_linear] = local_result[0];\
    }\
}
