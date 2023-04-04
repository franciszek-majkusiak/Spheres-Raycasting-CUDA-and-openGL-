#include "Scene.h"

__host__ void SetUpRandomSpheres(Scene& scene, int n, float maxRadius, float boxSize)
{
	SetUpRandom(scene.spheres, n, maxRadius, boxSize);
}

__host__ void SetUpRandomLights(Scene& scene, int n, float boxSize)
{
	SetUpRandom(scene.lights, n, boxSize);
}



__host__ void FreeCPU(Scene& scene)
{
	FreeCPU(scene.lights);
	FreeCPU(scene.spheres);
}



__host__ void Free(Scene& scene)
{
	Free(scene.lights);
	Free(scene.spheres);
}