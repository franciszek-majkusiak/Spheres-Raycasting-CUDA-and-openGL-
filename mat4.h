#pragma once

#include "cuda_runtime.h"


struct mat4
{
	float r1c1;
	float r1c2;
	float r1c3;
	float r1c4;
	float r2c1;
	float r2c2;
	float r2c3;
	float r2c4;
	float r3c1;
	float r3c2;
	float r3c3;
	float r3c4;
	float r4c1;
	float r4c2;
	float r4c3;
	float r4c4;

	__host__ __device__ mat4()
	{
		r1c1 = 0;
		r1c2 = 0;
		r1c3 = 0;
		r1c4 = 0;
		r2c1 = 0;
		r2c2 = 0;
		r2c3 = 0;
		r2c4 = 0;
		r3c1 = 0;
		r3c2 = 0;
		r3c3 = 0;
		r3c4 = 0;
		r4c1 = 0;
		r4c2 = 0;
		r4c3 = 0;
		r4c4 = 0;
	}


	//float r1c1, float r1c2, float r1c3, float r2c1, float r2c2, float r2c3, float r3c1, float r3c2, float r3c3
	//float r1c1, float r2c1, float r3c1, float r1c2, float r2c2, float r3c2, float r1c3, float r2c3, float r3c3
};

inline __host__ __device__ mat4 make_mat4(float r1c1, float r1c2, float r1c3, float r1c4, float r2c1, float r2c2, float r2c3, float r2c4, float r3c1, float r3c2, float r3c3, float r3c4, float r4c1, float r4c2, float r4c3, float r4c4)
{
	mat4 outMat;
	outMat.r1c1 = r1c1;
	outMat.r1c2 = r1c2;
	outMat.r1c3 = r1c3;
	outMat.r1c4 = r1c4;
	outMat.r2c1 = r2c1;
	outMat.r2c2 = r2c2;
	outMat.r2c3 = r2c3;
	outMat.r2c4 = r2c4;
	outMat.r3c1 = r3c1;
	outMat.r3c2 = r3c2;
	outMat.r3c3 = r3c3;
	outMat.r3c4 = r3c4;
	outMat.r4c1 = r4c1;
	outMat.r4c2 = r4c2;
	outMat.r4c3 = r4c3;
	outMat.r4c4 = r4c4;

	return outMat;
}

inline __host__ __device__ float3 operator*(mat4 mat, float3 vector)
{
	float4 vec = make_float4(vector.x, vector.y, vector.z, 1);
	vec = make_float4(mat.r1c1 * vec.x + mat.r1c2 * vec.y + mat.r1c3 * vec.z + mat.r1c4 * vec.w, mat.r2c1 * vec.x +mat.r2c2 * vec.y + mat.r2c3 * vec.z + mat.r2c4 * vec.w, mat.r3c1 * vec.x + mat.r3c2 * vec.y + mat.r3c3 * vec.z + mat.r3c4 * vec.w, mat.r4c1 * vec.x + mat.r4c2 * vec.y + mat.r4c3 * vec.z + mat.r4c4 * vec.w);
	return make_float3(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w);

}
