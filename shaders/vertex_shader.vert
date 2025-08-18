#version 450

//create a triangle using these 2D coordinates
vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    //draw our triangle
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}