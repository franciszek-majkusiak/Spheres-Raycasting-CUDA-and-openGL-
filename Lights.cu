#include "Lights.h"


__host__ void InitCPU(Lights& lights, int n)
{
	InitCPU(lights.position, n);
	InitCPU(lights.color, n);
	lights.i_a = new float[n];
	lights.i_d = new float[n];
	lights.i_s = new float[n];

	lights.n = new int[1];
	*lights.n = n;
	lights.h_n = n;
}

__host__ void FreeCPU(Lights& lights)
{
	FreeCPU(lights.position);
	FreeCPU(lights.color);
	delete[] lights.i_a;
	delete[] lights.i_d;
	delete[] lights.i_s;
	delete lights.n;
}

__host__ void Init(Lights& lights, int n)
{
	Init(lights.position, n);
	Init(lights.color, n);
	checkCudaErrors(cudaMalloc((void**)&lights.i_a, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&lights.i_d, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&lights.i_s, n * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&lights.n, sizeof(int)));
	checkCudaErrors(cudaMemcpy(lights.n, &n, sizeof(int), cudaMemcpyHostToDevice));
	lights.h_n = n;
}

__host__ void Free(Lights& lights)
{
	Free(lights.position);
	Free(lights.color);
	cudaFree(lights.i_a);
	cudaFree(lights.i_d);
	cudaFree(lights.i_s);
	cudaFree(lights.n);
}


__host__ void CopyHostToDevice(Lights& d_lights, Lights& h_lights)
{
	int n = h_lights.h_n;
	CopyHostToDevice(d_lights.position, h_lights.position, n);
	CopyHostToDevice(d_lights.color, h_lights.color, n);
	checkCudaErrors(cudaMemcpy(d_lights.i_a, h_lights.i_a, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_lights.i_d, h_lights.i_d, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_lights.i_s, h_lights.i_s, n * sizeof(float), cudaMemcpyHostToDevice));
}



__host__ void SetUpRandom(Lights& lights, int n, float boxSize)
{
	Lights newLights;
	InitCPU(newLights, n);
	if (lights.h_n != n)
	{
		if (lights.h_n != 0)
			Free(lights);
		Init(lights, n);
	}
	for (int i = 0; i < n; i++)
	{
		newLights.position.x[i] = RandomFloat(-boxSize / 2.0f, boxSize / 2.0f);
		newLights.position.y[i] = RandomFloat(-boxSize / 2.0f, boxSize / 2.0f);
		newLights.position.z[i] = RandomFloat(-boxSize / 2.0f, boxSize / 2.0f);

		newLights.color.x[i] = RandomFloat(0.0f, 1.0f);
		newLights.color.y[i] = RandomFloat(0.0f, 1.0f);
		newLights.color.z[i] = RandomFloat(0.0f, 1.0f);

		newLights.i_a[i] = RandomFloat(0.0f, 0.3f);
		newLights.i_d[i] = RandomFloat(0.3f, 1.0f);
		newLights.i_s[i] = RandomFloat(0.0f, 1.0f);
	}
	CopyHostToDevice(lights, newLights);
	FreeCPU(newLights);
}