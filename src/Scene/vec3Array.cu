#include "vec3Array.h"
#include <algorithm>
void InitCPU(vec3Arr& arr, int n)
{
	//printf("Init vec3Arr CPU\n");
	float* x;
	float* y;
	float* z;
	x = (float*)malloc(n * sizeof(float));
	y = (float*)malloc(n * sizeof(float));
	z = (float*)malloc(n * sizeof(float));

	arr.x = x;
	arr.y = y;
	arr.z = z;
}

void FreeCPU(vec3Arr& arr)
{
	//printf("Free vec3Arr CPU\n");
	free(arr.x);
	free(arr.y);
	free(arr.z);
}

void Swap(vec3Arr& arr1, vec3Arr& arr2)
{
	std::swap(arr1.x, arr2.x);
	std::swap(arr1.y, arr2.y);
	std::swap(arr1.z, arr2.z);
}



void Init(vec3Arr& arr, int n)
{
	//printf("Init GPU vec3Arr\n");
	float* x = 0;
	float* y = 0;
	float* z = 0;
	size_t size = n * sizeof(float);
	checkCudaErrors(cudaMalloc((void**)&x, size));
	checkCudaErrors(cudaMalloc((void**)&y, size));
	checkCudaErrors(cudaMalloc((void**)&z, size));

	arr.x = x;
	arr.y = y;
	arr.z = z;
}

void Free(vec3Arr& arr)
{
	//printf("Free GPU vec3Arr\n");
	cudaFree(arr.x);
	cudaFree(arr.y);
	cudaFree(arr.z);
}

void CopyHostToDevice(vec3Arr& d_arr, vec3Arr& h_arr, int n)
{
	checkCudaErrors(cudaMemcpy(d_arr.x, h_arr.x, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_arr.y, h_arr.y, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_arr.z, h_arr.z, n * sizeof(float), cudaMemcpyHostToDevice));
}

float RandomFloat(float min, float max)
{
	return (float)rand() / (float)RAND_MAX * (max - min) + min;
}