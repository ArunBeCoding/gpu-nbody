#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__ void bodyForce(Body *p, float dt, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
  float xi = p[i].x, yi = p[i].y, zi = p[i].z;

  for (int j = 0; j < n; j++) {
    float dx = p[j].x - xi;
    float dy = p[j].y - yi;
    float dz = p[j].z - zi;
    float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
    float invDist = 1.0f / sqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;

    Fx += dx * invDist3;
    Fy += dy * invDist3;
    Fz += dz * invDist3;
  }

  p[i].vx += dt * Fx;
  p[i].vy += dt * Fy;
  p[i].vz += dt * Fz;
}

int main(int argc, const char **argv) {
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f;
  const int nIters = 10;

  int bytes = nBodies * sizeof(Body);
  Body *p = (Body*)malloc(bytes);

  randomizeBodies((float*)p, 6 * nBodies);

  Body *d_p;
  cudaMalloc(&d_p, bytes);
  cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

  float totalTime = 0.0f;

  for (int iter = 1; iter <= nIters; iter++) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bodyForce<<<blocks, threadsPerBlock>>>(d_p, dt, nBodies);

    // integrate positions on GPU as well
    // or on CPU after copying (simpler):
    cudaMemcpy(p, d_p, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nBodies; i++) {
      p[i].x += p[i].vx * dt;
      p[i].y += p[i].vy * dt;
      p[i].z += p[i].vz * dt;
    }
    cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (iter > 1) totalTime += milliseconds;

    printf("Iteration %d: %.3f ms\n", iter, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  float avgTime = totalTime / (nIters - 1);
  printf("Average rate for iterations 2 through %d: %.3f ms per step\n",
         nIters, avgTime);
  printf("%d Bodies: average %.3f Billion Interactions / second\n",
         nBodies, 1e-6f * nBodies * nBodies / avgTime);

  cudaFree(d_p);
  free(p);
}

