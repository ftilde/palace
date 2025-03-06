#ifndef GLSL_PAGE_TABLE
#define GLSL_PAGE_TABLE

layout(buffer_reference) buffer Chunk;

#define PAGE_TABLE_LEVELS 3

#define BITS_PER_LEVEL 15
#define LEVEL_TABLE_SIZE (2 << BITS_PER_LEVEL)
#define LEVEL_TABLE_MASK (LEVEL_TABLE_SIZE - 1)

#define LEVEL_IDENTIFIER_BITS 2
#define LEVEL_IDENTIFIER_SIZE (2 << LEVEL_IDENTIFIER_BITS)
#define LEVEL_IDENTIFIER_MASK (LEVEL_IDENTIFIER_SIZE - 1)

#define PAGE_TABLE_INDEX_UNUSED 0xffffffffffffffffL

layout(buffer_reference, std430) buffer PageTablePage {
    uint64_t page_table_index;
    uint64_t values[LEVEL_TABLE_SIZE];
};

uint level(uint64_t page_table_index) {
    return uint((page_table_index >> (3*BITS_PER_LEVEL)) & LEVEL_TABLE_MASK);
}

uint64_t with_level(uint64_t chunk_index, uint level) {
    uint64_t level_bits = uint64_t(level) << (3*BITS_PER_LEVEL);
    uint64_t masked = (chunk_index >> (level * BITS_PER_LEVEL)) << (level_bits * BITS_PER_LEVEL);

    return level_bits | masked;
}

uvec3 page_table_index_to_level_indices(uint64_t page_table_index) {
    return uvec3(
        uint((page_table_index >> (2*BITS_PER_LEVEL)) & LEVEL_TABLE_MASK),
        uint((page_table_index >> (1*BITS_PER_LEVEL)) & LEVEL_TABLE_MASK),
        uint( page_table_index                        & LEVEL_TABLE_MASK)
    );
}

#endif //GLSL_PAGE_TABLE
