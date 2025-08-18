#version 450

layout(location = 0) out vec4 outColor;

void main() {
    //set the triangles color to red, fully opaque
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}