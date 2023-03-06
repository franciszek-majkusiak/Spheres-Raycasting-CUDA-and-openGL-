#version 330 core
out vec4 FragColor;
  
uniform sampler2D FrameToDraw;

in vec2 TexCoords;

void main()
{
    FragColor = texture(FrameToDraw, TexCoords);
} 