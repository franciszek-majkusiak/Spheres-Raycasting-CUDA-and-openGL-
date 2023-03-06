#pragma once
#include <device_launch_parameters.h>
#include "helper_math.h"
#include <stdio.h>
#include <cstdlib>


#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)



struct vec3Arr
{
	float* x = nullptr;
	float* y = nullptr;
	float* z = nullptr;
};


inline __host__ __device__ float3 GetValue(vec3Arr& arr, int i)
{
	return make_float3(arr.x[i], arr.y[i], arr.z[i]);
}

inline __host__ __device__ void SetValue(vec3Arr& arr, int i, float3 value)
{
	arr.x[i] = value.x;
	arr.y[i] = value.y;
	arr.z[i] = value.z;
}



void InitCPU(vec3Arr& arr, int n);
void FreeCPU(vec3Arr& arr);
void Init(vec3Arr& arr, int n);
void Free(vec3Arr& arr);
void Swap(vec3Arr& arr1, vec3Arr& arr2);
void CopyHostToDevice(vec3Arr& d_arr, vec3Arr& h_arr, int n);




float RandomFloat(float min, float max);