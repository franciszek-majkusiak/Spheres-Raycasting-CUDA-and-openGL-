#pragma once


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <device_launch_parameters.h>
#include "helper_math.h"

#include "glm/glm.hpp"
#include "Helpers/mat3.h"
#include "Scene/Scene.h"

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

namespace Renderer
{
    #define EPSILON 1e-8f

    void launch_renderSpheres(int width, int height, unsigned int* imageData, Scene& scene, glm::vec3 cameraPosition, mat3 fur, float tanHalfFov);

}
