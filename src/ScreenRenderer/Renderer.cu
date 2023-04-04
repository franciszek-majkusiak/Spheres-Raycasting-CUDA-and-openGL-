#include "Renderer.h"
#include <glm/gtc/matrix_access.hpp>

namespace Renderer
{

	__host__ __device__ bool RayIntersectsSphere(float3 rayOrigin, float3 rayDirection, float3 center, float r, float3& intersectionPoint, float3& normal)
	{
		float t0, t1;
		float rSQ = r * r;
		float3 L = center - rayOrigin;
		float tca = dot(L, rayDirection);
		float d2 = dot(L, L) - tca * tca;
		if (d2 > rSQ) return false;
		float thc = sqrt(rSQ - d2);
		t0 = tca - thc;
		t1 = tca + thc;
		if (t0 > t1)
		{
			float temp = t1;
			t1 = t0;
			t0 = temp;
		}

		if (t0 < 0)
		{
			t0 = t1;
			if (t0 < EPSILON) return false;
		}

		intersectionPoint = rayOrigin + rayDirection * t0;
		normal = normalize(intersectionPoint - center);

		return true;
	}

	__device__ float clamp(float x, float a, float b)
	{
		return max(a, min(b, x));
	}

	__device__ int clamp(int x, int a, int b)
	{
		return  max(a, min(b, x));
	}

	
	__device__ int rgbToInt(float r, float g, float b)
	{
		r = clamp(r, 0.0f, 255.0f);
		g = clamp(g, 0.0f, 255.0f);
		b = clamp(b, 0.0f, 255.0f);
		return (int(b) << 16) | (int(g) << 8) | int(r);
	}


	__global__ void renderSpheres(unsigned int* imageData, Scene scene, int width, int height, float3 cameraPosition, float tanHalfFov, mat3 fur)
	{

		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int bw = blockDim.x;
		int bh = blockDim.y;
		int x = blockIdx.x * bw + tx;
		int y = blockIdx.y * bh + ty;
		if (x > width || y > height)
			return;



		float aspectRatio = (float)width / (float)height;
		float imageHeight = tanHalfFov;
		float imageWidth = imageHeight * aspectRatio;

		float xPercentage = (float)x / (float)width;
		float yPercentage = (float)y / (float)height;

		float xI = -imageWidth / 2 + imageWidth * xPercentage;
		float yI = -imageHeight / 2 + imageHeight * yPercentage;


		float3 front = make_float3(fur.r1c1, fur.r1c2, fur.r1c3);
		float3 up = make_float3(fur.r2c1, fur.r2c2, fur.r2c3);
		float3 right = make_float3(fur.r3c1, fur.r3c2, fur.r3c3);
		float3 worldPos = cameraPosition + front + up * yI + right * xI;

		float3 rayDirection = normalize(worldPos - cameraPosition);

		float3 hitPoint, normal;

		float zBuff = 999999.0f;

		imageData[y * width + x] = rgbToInt(255.0f, 255.0f, 255.0f);
		int numOfSpheres = *scene.spheres.n;
		for (int i = 0; i < numOfSpheres; i++)
		{
			float3 center = GetValue(scene.spheres.center, i);
			float r = scene.spheres.radius[i];
			bool hit = RayIntersectsSphere(cameraPosition, rayDirection, center, r, hitPoint, normal);
			if (hit)
			{
				float dist = length(hitPoint - cameraPosition);
				if (zBuff > dist)
				{
					zBuff = dist;
					int n = *scene.lights.n;
					float3 outColor = make_float3(0.0f, 0.0f, 0.0f);
					float3 sphereColor = GetValue(scene.spheres.color, i);
					float k_a = scene.spheres.k_a[i];
					float k_d = scene.spheres.k_d[i];
					float k_s = scene.spheres.k_s[i];

					float constant = 1.0f;
					float linear = 0.014;
					float quadratic = 0.0007;

					for (int j = 0; j < n; j++)
					{
						float3 lightPos = GetValue(scene.lights.position, j);
						float3 lightColor = GetValue(scene.lights.color, j);
						float distance = length(hitPoint - lightPos);
						float attenuation = 1.0 / (constant + linear * distance + quadratic * distance * distance);
						//lightColor *= attenuation;
						float i_a = scene.lights.i_a[j];
						float i_d = scene.lights.i_d[j];
						float i_s = scene.lights.i_s[j];

						float3 L = normalize(lightPos - hitPoint);
						float3 V = normalize(cameraPosition - hitPoint);
						float3 R = 2.0f * dot(normal, L) * (normal - L);
						float cosNL = max(dot(normal, L), 0.0f);
						float cosVR = max(dot(V, R), 0.0f);

						float3 curColor = i_a * k_a + i_d * k_d * lightColor * sphereColor * cosNL + i_s * k_s * lightColor * sphereColor * cosVR * cosVR;
						curColor *= 255.0f;
						outColor += make_float3(clamp(curColor.x, 0.0f, 255.0f), clamp(curColor.y, 0.0f, 255.0f), clamp(curColor.z, 0.0f, 255.0f));

					}
					imageData[y * width + x] = rgbToInt(outColor.x, outColor.y, outColor.z);
				}
			}
		}
	}


	void launch_renderSpheres(int width, int height, unsigned int* imageData, Scene& scene, glm::vec3 cameraPosition, mat3 fur, float tanHalfFov)
	{
		dim3 block(32, 16, 1);
		dim3 grid(ceil((float)width / (float)block.x), ceil((float)height / (float)block.y), 1);

		float3 cameraPos = make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);


		renderSpheres << < grid, block >> > (imageData, scene, width, height, cameraPos, tanHalfFov, fur);

		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			exit(EXIT_FAILURE);
		}
	}

}
