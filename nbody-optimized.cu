#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 256

// Structure of Arrays for better memory coalescing
typedef struct {
  float *x, *y, *z;     // positions
  float *vx, *vy, *vz;  // velocities
} BodySoA;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

// Allocate Structure of Arrays
void allocateBodies(BodySoA *bodies, int n) {
  cudaMalloc(&bodies->x, n * sizeof(float));
  cudaMalloc(&bodies->y, n * sizeof(float));
  cudaMalloc(&bodies->z, n * sizeof(float));
  cudaMalloc(&bodies->vx, n * sizeof(float));
  cudaMalloc(&bodies->vy, n * sizeof(float));
  cudaMalloc(&bodies->vz, n * sizeof(float));
}

void freeBodies(BodySoA *bodies) {
  cudaFree(bodies->x);
  cudaFree(bodies->y);
  cudaFree(bodies->z);
  cudaFree(bodies->vx);
  cudaFree(bodies->vy);
  cudaFree(bodies->vz);
}

// Optimized force calculation with shared memory tiling
__global__ void bodyForce(BodySoA bodies, float dt, int n) {
  // Shared memory for tiling
  __shared__ float shPosX[BLOCK_SIZE];
  __shared__ float shPosY[BLOCK_SIZE];
  __shared__ float shPosZ[BLOCK_SIZE];
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
  float xi, yi, zi;
  
  // Load this body's position
  if (i < n) {
    xi = bodies.x[i];
    yi = bodies.y[i];
    zi = bodies.z[i];
  }
  
  // Tile across all bodies
  int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  for (int tile = 0; tile < numTiles; tile++) {
    int idx = tile * BLOCK_SIZE + threadIdx.x;
    
    // Load tile into shared memory (coalesced)
    if (idx < n) {
      shPosX[threadIdx.x] = bodies.x[idx];
      shPosY[threadIdx.x] = bodies.y[idx];
      shPosZ[threadIdx.x] = bodies.z[idx];
    }
    __syncthreads();
    
    // Compute forces with bodies in this tile
    if (i < n) {
      #pragma unroll 8
      for (int j = 0; j < BLOCK_SIZE; j++) {
        int bodyIdx = tile * BLOCK_SIZE + j;
        if (bodyIdx < n) {
          float dx = shPosX[j] - xi;
          float dy = shPosY[j] - yi;
          float dz = shPosZ[j] - zi;
          float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
          
          // Use fast inverse square root
          float invDist = rsqrtf(distSqr);
          float invDist3 = invDist * invDist * invDist;
          
          Fx += dx * invDist3;
          Fy += dy * invDist3;
          Fz += dz * invDist3;
        }
      }
    }
    __syncthreads();
  }
  
  // Update velocities (write coalesced)
  if (i < n) {
    bodies.vx[i] += dt * Fx;
    bodies.vy[i] += dt * Fy;
    bodies.vz[i] += dt * Fz;
  }
}

// Integrate positions on GPU (no CPU transfer!)
__global__ void integratePositions(BodySoA bodies, float dt, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    bodies.x[i] += bodies.vx[i] * dt;
    bodies.y[i] += bodies.vy[i] * dt;
    bodies.z[i] += bodies.vz[i] * dt;
  }
}

int main(int argc, const char **argv) {
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f;
  const int nIters = 10;

  // Host arrays for initialization
  float *h_x = (float*)malloc(nBodies * sizeof(float));
  float *h_y = (float*)malloc(nBodies * sizeof(float));
  float *h_z = (float*)malloc(nBodies * sizeof(float));
  float *h_vx = (float*)malloc(nBodies * sizeof(float));
  float *h_vy = (float*)malloc(nBodies * sizeof(float));
  float *h_vz = (float*)malloc(nBodies * sizeof(float));

  // Initialize with random values
  randomizeBodies(h_x, nBodies);
  randomizeBodies(h_y, nBodies);
  randomizeBodies(h_z, nBodies);
  randomizeBodies(h_vx, nBodies);
  randomizeBodies(h_vy, nBodies);
  randomizeBodies(h_vz, nBodies);

  // Allocate device memory (SoA)
  BodySoA d_bodies;
  allocateBodies(&d_bodies, nBodies);

  // Copy to device
  cudaMemcpy(d_bodies.x, h_x, nBodies * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bodies.y, h_y, nBodies * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bodies.z, h_z, nBodies * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bodies.vx, h_vx, nBodies * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bodies.vy, h_vy, nBodies * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bodies.vz, h_vz, nBodies * sizeof(float), cudaMemcpyHostToDevice);

  int blocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Create events outside loop for better performance
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float totalTime = 0.0f;

  printf("Running %d iterations with %d bodies...\n", nIters, nBodies);

  for (int iter = 1; iter <= nIters; iter++) {
    cudaEventRecord(start);

    // Compute forces and update velocities
    bodyForce<<<blocks, BLOCK_SIZE>>>(d_bodies, dt, nBodies);
    
    // Integrate positions on GPU (no CPU copy!)
    integratePositions<<<blocks, BLOCK_SIZE>>>(d_bodies, dt, nBodies);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (iter > 1) totalTime += milliseconds;

    printf("Iteration %d: %.3f ms\n", iter, milliseconds);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float avgTime = totalTime / (nIters - 1);
  printf("\n=== Performance Summary ===\n");
  printf("Average time (iterations 2-%d): %.3f ms per step\n", nIters, avgTime);
  printf("%d Bodies: average %.3f Billion Interactions / second\n",
         nBodies, 1e-6f * nBodies * nBodies / avgTime);
  
  double gflops = 1e-9 * nBodies * nBodies * 20.0 / (avgTime / 1000.0);
  printf("Computational throughput: %.2f GFLOPS\n", gflops);

  // Optional: Copy final positions back for validation
  // cudaMemcpy(h_x, d_bodies.x, nBodies * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  freeBodies(&d_bodies);
  free(h_x); free(h_y); free(h_z);
  free(h_vx); free(h_vy); free(h_vz);

  return 0;
}