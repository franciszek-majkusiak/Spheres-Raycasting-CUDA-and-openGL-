#pragma once

#include "vec3Array.h"
#include <cstdlib>


struct Spheres
{
	int* n = nullptr;
	int h_n = 0;
	vec3Arr center;
	vec3Arr color;
	float* k_a = nullptr;
	float* k_d = nullptr;
	float* k_s = nullptr;
	float* radius = nullptr;
};

__host__ void InitCPU(Spheres& spheres, int n);


__host__ void Init(Spheres& spheres, int n);

__host__ void FreeCPU(Spheres& spheres);



__host__ void Free(Spheres& spheres);

__host__ void CopyHostToDevice(Spheres& d_spheres, Spheres& h_spheres);


__host__ void SetUpRandom(Spheres& spheres, int n, float maxRadius = 3.0f, float boxSize = 1000.0f);
