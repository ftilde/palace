u8vec4 from_uniform(vec4 v) {
    return u8vec4(v * 255);
}

vec4 to_uniform(u8vec4 v) {
    // TODO: Why the hell do we need to clamp here? Fixes an overflow bug in DVR, though
    return min(vec4(1.0), vec4(v)/255.0);
}

u8vec3 from_uniform(vec3 v) {
    return u8vec3(v * 255);
}

vec3 to_uniform(u8vec3 v) {
    // TODO: Why the hell do we need to clamp here? Fixes an overflow bug in DVR, though
    return min(vec3(1.0), vec3(v)/255.0);
}

u8vec4 intensity_to_grey(float v) {
    v = clamp(v, 0.0, 1.0);
    return from_uniform(vec4(v, v, v, 1));
}

#define apply_tf(tf_table, tf_table_len, tf_min, tf_max, input_val, result) {\
    float norm = ((input_val)-(tf_min))/((tf_max )- (tf_min));\
    uint index = min(uint(max(0.0, norm) * (tf_table_len)), (tf_table_len) - 1);\
    (result) = tf_table[index];\
}
