#version 450

layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec4 a_Color;

layout(location = 0) out vec4 f_Color;
layout(location = 1) out flat int discarding;

layout(set=0, binding=0) uniform Uniforms {
    mat4x4 u_view_proj;
    vec2 offset;
    vec2 size;
    vec2 window_size;
};

struct RemappingResult {
    float pos;
    int discarding;
};

RemappingResult remap(float in_value, float offset, float window_size, float size) {
    float point = offset / window_size;
    point = point * 2.0f - 1.0f;
    float start = point;
    float end = start + 2.0f * size / window_size;
    float coef = (end - start) / 2.0f;
    float b = start + (end - start) / 2.0f;
    RemappingResult result;
    result.pos = b + coef * in_value;
    result.discarding = (result.pos >= start && result.pos <= end) ? 0 : 1;
    return result;
}

void main() {
    f_Color = a_Color;
    vec4 projected = u_view_proj * vec4(a_Pos, 1.0);
    vec3 point = projected.xyz / projected.w;
    RemappingResult remapping_x = remap(point.x, offset.x, window_size.x, size.x);
    RemappingResult remapping_y = remap(point.y, offset.y, window_size.y, size.y);

    point.x = remapping_x.pos;
    point.y = remapping_y.pos * -1.0f;

    discarding = remapping_x.discarding == 0 ? (remapping_y.discarding == 0 ? 0 : 1) : 1;

    gl_Position = vec4(point, 1.0f);
    gl_PointSize = 5.0;
}