#include "Screen.cuh"
#include "stb_image.h"
#include "Renderer.h"

void Screen::Init(int scrWidth, int scrHeight)
{
    this->width = scrWidth;
    this->height = scrHeight;

    shader = Shader("texture.vert", "texture.frag");
    shader.Activate();
    shader.setInt("FrameToDraw", 0);

    float vertices[] = {
       -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
       1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
       1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
       -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    };

    int indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);


    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &screenGLTexture);

    glBindTexture(GL_TEXTURE_2D, screenGLTexture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsGLRegisterImage(&screenCudaResource, screenGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    size_t size = sizeof(GLubyte) * 4 * width * height;
    cudaMalloc((void**)&renderBuffer, size);
}





void Screen::RenderSpheres(Scene& scene, glm::vec3 cameraPosition, mat3 fur, float tanHalfFov)
{
    Renderer::launch_renderSpheres(width, height, renderBuffer, scene, cameraPosition, fur, tanHalfFov);


    cudaArray* texture_ptr;
    cudaGraphicsMapResources(1, &screenCudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, screenCudaResource, 0, 0);

    size_t size = sizeof(GLubyte) * 4 * width * height;
    cudaMemcpyToArray(texture_ptr, 0, 0, renderBuffer, size, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &screenCudaResource, 0);



    shader.Activate();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, screenGLTexture);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

Screen::~Screen()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}


void Screen::Resize(int width, int height)
{
    this->width = width;
    this->height = height;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &screenGLTexture);

    glBindTexture(GL_TEXTURE_2D, screenGLTexture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsGLRegisterImage(&screenCudaResource, screenGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    cudaFree(renderBuffer);
    size_t size = sizeof(GLubyte) * 4 * width * height;
    cudaMalloc((void**)&renderBuffer, size);
}

