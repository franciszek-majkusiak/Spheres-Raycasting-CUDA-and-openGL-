#pragma once
#include "vec3Array.h"

struct Lights
{
	int* n = nullptr;
	int h_n = 0;
	vec3Arr position;

	vec3Arr color;
	float* i_a = nullptr;
	float* i_d = nullptr;
	float* i_s = nullptr;
};

__host__ void InitCPU(Lights& lights, int n);

__host__ void FreeCPU(Lights& lights);

__host__ void Init(Lights& lights, int n);

__host__ void Free(Lights& lights);


__host__ void CopyHostToDevice(Lights& d_lights, Lights& h_lights);

__host__ void SetUpRandom(Lights& lights, int n, float boxSize = 1000.0f);