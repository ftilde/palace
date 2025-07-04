#ifndef GLSL_HASH
#define GLSL_HASH
// Extracted from murmur3 (public domain) (not used)
//uint hash(uint h) {
//  h ^= h >> 16;
//  h *= 0x85ebca6b;
//  h ^= h >> 13;
//  h *= 0xc2b2ae35;
//  h ^= h >> 16;
//
//  return h;
//}
// 64 bit version
uint hash(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdUL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53UL;
    k ^= k >> 33;

    return uint(k);
}

#define MAX_TRIES 5
#define EMPTY 0xffffffffffffffffL
#define try_insert_into_hash_table(table, size_expr, value_expr) {\
    uint64_t value = (value_expr);\
    uint size = (size_expr);\
    uint pos = hash(value)%size;\
    /* Perform linear probing with a maximum of MAX_TRIES tries*/\
    for(uint i = 0; i<MAX_TRIES; ++i) {\
        uint cell_index = (pos+i) % size;\
        uint64_t replaced = atomicCompSwap(table[cell_index], EMPTY, value);\
        if(replaced == EMPTY || replaced == value) {\
            /* The insert was either successful or already present at the location, so we are done.*/\
            break;\
        }\
    }\
}
#endif //GLSL_HASH
