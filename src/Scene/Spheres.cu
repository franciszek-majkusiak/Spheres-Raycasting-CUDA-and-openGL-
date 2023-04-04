#include "Spheres.h"



__host__ void InitCPU(Spheres& spheres, int n)
{
	spheres.n = new int;
	*spheres.n = n;
	spheres.h_n = n;
	InitCPU(spheres.center, n);
	InitCPU(spheres.color, n);
	spheres.radius = new float[n];
	spheres.k_a = new float[n];
	spheres.k_d = new float[n];
	spheres.k_s = new float[n];
}


__host__ void Init(Spheres& spheres, int n)
{
	spheres.h_n = n;
	Init(spheres.center, n);
	Init(spheres.color, n);
	checkCudaErrors(cudaMalloc((void**)&spheres.radius, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&spheres.k_a, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&spheres.k_d, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&spheres.k_s, n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&spheres.n, sizeof(int)));
	checkCudaErrors(cudaMemcpy(spheres.n, &n, sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void FreeCPU(Spheres& spheres)
{
	FreeCPU(spheres.center);
	FreeCPU(spheres.color);
	delete[] spheres.radius;
	delete[] spheres.k_a;
	delete[] spheres.k_d;
	delete[] spheres.k_s;
	delete spheres.n;
}



__host__ void Free(Spheres& spheres)
{
	Free(spheres.center);
	Free(spheres.color);
	cudaFree(spheres.radius);
	cudaFree(spheres.k_a);
	cudaFree(spheres.k_d);
	cudaFree(spheres.k_s);
	cudaFree(spheres.n);
}

__host__ void CopyHostToDevice(Spheres& d_spheres, Spheres& h_spheres)
{
	int n = h_spheres.h_n;
	checkCudaErrors(cudaMemcpy(d_spheres.radius, h_spheres.radius, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_spheres.k_a, h_spheres.k_a, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_spheres.k_d, h_spheres.k_d, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_spheres.k_s, h_spheres.k_s, n * sizeof(float), cudaMemcpyHostToDevice));
	CopyHostToDevice(d_spheres.center, h_spheres.center, n);
	CopyHostToDevice(d_spheres.color, h_spheres.color, n);
}



__host__ void SetUpRandom(Spheres& spheres, int n, float maxRadius, float boxSize)
{
	Spheres newSpheres;
	InitCPU(newSpheres, n);
	if (spheres.h_n != n)
	{
		if(spheres.h_n != 0)
			Free(spheres);
		Init(spheres, n);
	}
	for (int i = 0; i < n; i++)
	{
		newSpheres.radius[i] = RandomFloat(0.0f, maxRadius);

		newSpheres.center.x[i] = RandomFloat(-boxSize / 2.0f, boxSize / 2.0f);
		newSpheres.center.y[i] = RandomFloat(-boxSize / 2.0f, boxSize / 2.0f);
		newSpheres.center.z[i] = RandomFloat(-boxSize / 2.0f, boxSize / 2.0f);

		newSpheres.color.x[i] = RandomFloat(0.0f, 1.0f);
		newSpheres.color.y[i] = RandomFloat(0.0f, 1.0f);
		newSpheres.color.z[i] = RandomFloat(0.0f, 1.0f);

		newSpheres.k_a[i] = RandomFloat(0.0f, 0.3f);
		newSpheres.k_d[i] = RandomFloat(0.3f, 1.0f);
		newSpheres.k_s[i] = RandomFloat(0.0f, 1.0f);
	}
	CopyHostToDevice(spheres, newSpheres);
	FreeCPU(newSpheres);
}