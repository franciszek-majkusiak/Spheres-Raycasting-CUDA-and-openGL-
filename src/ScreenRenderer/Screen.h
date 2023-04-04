#pragma once
#include <glad/glad.h>
#include "cuda_gl_interop.h"
#include <iostream>
#include "Shaders/Shader.h"
#include "Helpers/mat3.h"
#include "Scene/Scene.h"


class Screen
{
public:
	void Init(int width, int height);
	~Screen();
	int width;
	int height;
	void RenderSpheres(Scene& scene, glm::vec3 cameraPosition, mat3 fur, float tanHalfFov);
	void Resize(int width, int height);
private:
	GLuint screenGLTexture;
	cudaGraphicsResource_t screenCudaResource;
	unsigned int* renderBuffer;
	GLuint VAO, VBO, EBO;
	Shader shader;
};
