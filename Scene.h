#pragma once
#include "Lights.h"
#include "Spheres.h"
#include <vector>
#include <glm/glm.hpp>

struct Scene
{
	Lights lights;
	Spheres spheres;
};





__host__ void SetUpRandomSpheres(Scene& scene, int n, float maxRadius = 3.0f, float boxSize = 1000.0f);

__host__ void SetUpRandomLights(Scene& scene, int n, float boxSize = 1000.0f);



__host__ void FreeCPU(Scene& scene);



__host__ void Free(Scene& scene);