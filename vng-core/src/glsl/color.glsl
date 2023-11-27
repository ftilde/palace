u8vec4 from_uniform(vec4 v) {
    return u8vec4(v * 255);
}

vec4 to_uniform(u8vec4 v) {
    return vec4(v)/255.0;
}

u8vec4 intensity_to_grey(float v) {
    v = clamp(v, 0.0, 1.0);
    return from_uniform(vec4(v, v, v, 1));
}
