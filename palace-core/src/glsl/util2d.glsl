#ifndef UTIL2D_GLSL
#define UTIL2D_GLSL
#define COLOR_CHECKERED_DARK u8vec4(160, 160, 160, 0);
#define COLOR_CHECKERED_LIGHT u8vec4(96, 96, 96, 0);
#define COLOR_NOT_LOADED u8vec4(0, 0, 0, 0);

#define CHECKERED_PATTERN_SIZE_PIXELS 32;

u8vec4 checkered_color(uvec2 pos) {
    uvec2 check_pos = pos / CHECKERED_PATTERN_SIZE_PIXELS;
    uvec2 check_even = check_pos % 2;

    if(check_even.x == check_even.y) {
        return COLOR_CHECKERED_DARK;
    } else {
        return COLOR_CHECKERED_LIGHT;
    }
}
#endif
