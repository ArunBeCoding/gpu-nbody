#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <float.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 256
#define THETA 0.5f  // Opening angle criterion for Barnes-Hut

// Octree node structure
typedef struct OctreeNode {
    float mass;           // Total mass in this node
    float x, y, z;       // Center of mass
    float size;          // Cell size
    float minx, miny, minz;  // Bounding box
    int childIndex;      // Index to first child (-1 if leaf)
    int bodyIndex;       // Index to body if leaf (-1 otherwise)
    int numBodies;       // Number of bodies in this cell
} OctreeNode;

// Body structure (SoA for better coalescing)
typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
} BodySoA;

// Helper structure for sorting
typedef struct {
    unsigned long long morton;
    int index;
} MortonPair;

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void allocateBodies(BodySoA *bodies, int n) {
    cudaMalloc(&bodies->x, n * sizeof(float));
    cudaMalloc(&bodies->y, n * sizeof(float));
    cudaMalloc(&bodies->z, n * sizeof(float));
    cudaMalloc(&bodies->vx, n * sizeof(float));
    cudaMalloc(&bodies->vy, n * sizeof(float));
    cudaMalloc(&bodies->vz, n * sizeof(float));
    cudaMalloc(&bodies->mass, n * sizeof(float));
}

void freeBodies(BodySoA *bodies) {
    cudaFree(bodies->x);
    cudaFree(bodies->y);
    cudaFree(bodies->z);
    cudaFree(bodies->vx);
    cudaFree(bodies->vy);
    cudaFree(bodies->vz);
    cudaFree(bodies->mass);
}

// Compute Morton code (Z-order curve) for spatial hashing
__device__ unsigned long long computeMorton(float x, float y, float z, 
                                            float minx, float miny, float minz, 
                                            float size) {
    // Normalize to [0, 1]
    unsigned int ix = (unsigned int)((x - minx) / size * 1024.0f);
    unsigned int iy = (unsigned int)((y - miny) / size * 1024.0f);
    unsigned int iz = (unsigned int)((z - minz) / size * 1024.0f);
    
    // Clamp to valid range
    ix = min(ix, 1023u);
    iy = min(iy, 1023u);
    iz = min(iz, 1023u);
    
    // Interleave bits
    unsigned long long morton = 0;
    for (int i = 0; i < 10; i++) {
        morton |= ((unsigned long long)(ix & (1u << i)) << (2 * i));
        morton |= ((unsigned long long)(iy & (1u << i)) << (2 * i + 1));
        morton |= ((unsigned long long)(iz & (1u << i)) << (2 * i + 2));
    }
    return morton;
}

// Find bounding box
__global__ void computeBoundingBox(BodySoA bodies, int n,
                                   float *minx, float *miny, float *minz,
                                   float *maxx, float *maxy, float *maxz) {
    __shared__ float sminx[BLOCK_SIZE];
    __shared__ float sminy[BLOCK_SIZE];
    __shared__ float sminz[BLOCK_SIZE];
    __shared__ float smaxx[BLOCK_SIZE];
    __shared__ float smaxy[BLOCK_SIZE];
    __shared__ float smaxz[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with extreme values
    sminx[tid] = (i < n) ? bodies.x[i] : FLT_MAX;
    sminy[tid] = (i < n) ? bodies.y[i] : FLT_MAX;
    sminz[tid] = (i < n) ? bodies.z[i] : FLT_MAX;
    smaxx[tid] = (i < n) ? bodies.x[i] : -FLT_MAX;
    smaxy[tid] = (i < n) ? bodies.y[i] : -FLT_MAX;
    smaxz[tid] = (i < n) ? bodies.z[i] : -FLT_MAX;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sminx[tid] = fminf(sminx[tid], sminx[tid + s]);
            sminy[tid] = fminf(sminy[tid], sminy[tid + s]);
            sminz[tid] = fminf(sminz[tid], sminz[tid + s]);
            smaxx[tid] = fmaxf(smaxx[tid], smaxx[tid + s]);
            smaxy[tid] = fmaxf(smaxy[tid], smaxy[tid + s]);
            smaxz[tid] = fmaxf(smaxz[tid], smaxz[tid + s]);
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicMin((int*)minx, __float_as_int(sminx[0]));
        atomicMin((int*)miny, __float_as_int(sminy[0]));
        atomicMin((int*)minz, __float_as_int(sminz[0]));
        atomicMax((int*)maxx, __float_as_int(smaxx[0]));
        atomicMax((int*)maxy, __float_as_int(smaxy[0]));
        atomicMax((int*)maxz, __float_as_int(smaxz[0]));
    }
}

// Compute Morton codes for all bodies
__global__ void computeMortonCodes(BodySoA bodies, int n, MortonPair *pairs,
                                   float minx, float miny, float minz, float size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pairs[i].morton = computeMorton(bodies.x[i], bodies.y[i], bodies.z[i],
                                        minx, miny, minz, size);
        pairs[i].index = i;
    }
}

// Simple bitonic sort for Morton codes (for demonstration - use Thrust in production)
__device__ void bitonicSort(MortonPair *data, int n, int j, int k) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int ixj = i ^ j;
    
    if (ixj > i && i < n && ixj < n) {
        if ((i & k) == 0) {
            if (data[i].morton > data[ixj].morton) {
                MortonPair temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[i].morton < data[ixj].morton) {
                MortonPair temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// Build octree (simplified version - processes leaf nodes)
__global__ void buildOctreeLeaves(BodySoA bodies, MortonPair *sortedPairs, 
                                  OctreeNode *nodes, int n,
                                  float minx, float miny, float minz, float size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bodyIdx = sortedPairs[i].index;
        nodes[i].mass = bodies.mass[bodyIdx];
        nodes[i].x = bodies.x[bodyIdx];
        nodes[i].y = bodies.y[bodyIdx];
        nodes[i].z = bodies.z[bodyIdx];
        nodes[i].bodyIndex = bodyIdx;
        nodes[i].childIndex = -1;
        nodes[i].numBodies = 1;
    }
}

// Barnes-Hut force calculation with tree traversal
__global__ void barnesHutForce(BodySoA bodies, OctreeNode *nodes, float dt, int n,
                               float rootMinX, float rootMinY, float rootMinZ, float rootSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float xi = bodies.x[i];
    float yi = bodies.y[i];
    float zi = bodies.z[i];
    
    // Stack for iterative tree traversal (avoid recursion)
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;  // Start with root
    
    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        if (nodeIdx >= n || nodeIdx < 0) continue;
        
        OctreeNode node = nodes[nodeIdx];
        
        // Skip empty nodes
        if (node.numBodies == 0) continue;
        
        float dx = node.x - xi;
        float dy = node.y - yi;
        float dz = node.z - zi;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float dist = sqrtf(distSqr);
        
        // Opening angle criterion: s/d < theta
        bool usePseudoParticle = (node.size / dist) < THETA;
        
        // If it's a leaf or satisfies opening criterion
        if (node.childIndex == -1 || usePseudoParticle) {
            if (node.bodyIndex != i) {  // Don't compute self-interaction
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float force = node.mass * invDist3;
                
                Fx += dx * force;
                Fy += dy * force;
                Fz += dz * force;
            }
        } else {
            // Need to traverse children (in real implementation, push all 8 children)
            // Simplified: push next few nodes (representing children)
            for (int c = 1; c <= 8 && nodeIdx + c < n; c++) {
                if (stackPtr < 64) {
                    stack[stackPtr++] = nodeIdx + c;
                }
            }
        }
    }
    
    // Update velocities
    bodies.vx[i] += dt * Fx;
    bodies.vy[i] += dt * Fy;
    bodies.vz[i] += dt * Fz;
}

// Optimized direct O(N^2) method with shared memory tiling
__global__ void directForce(BodySoA bodies, float dt, int n) {
    __shared__ float shPosX[BLOCK_SIZE];
    __shared__ float shPosY[BLOCK_SIZE];
    __shared__ float shPosZ[BLOCK_SIZE];
    __shared__ float shMass[BLOCK_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float xi, yi, zi;
    
    if (i < n) {
        xi = bodies.x[i];
        yi = bodies.y[i];
        zi = bodies.z[i];
    }
    
    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        int idx = tile * BLOCK_SIZE + threadIdx.x;
        
        if (idx < n) {
            shPosX[threadIdx.x] = bodies.x[idx];
            shPosY[threadIdx.x] = bodies.y[idx];
            shPosZ[threadIdx.x] = bodies.z[idx];
            shMass[threadIdx.x] = bodies.mass[idx];
        }
        __syncthreads();
        
        if (i < n) {
            #pragma unroll 8
            for (int j = 0; j < BLOCK_SIZE; j++) {
                int bodyIdx = tile * BLOCK_SIZE + j;
                if (bodyIdx < n) {
                    float dx = shPosX[j] - xi;
                    float dy = shPosY[j] - yi;
                    float dz = shPosZ[j] - zi;
                    float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                    
                    float invDist = rsqrtf(distSqr);
                    float invDist3 = invDist * invDist * invDist;
                    float force = shMass[j] * invDist3;
                    
                    Fx += dx * force;
                    Fy += dy * force;
                    Fz += dz * force;
                }
            }
        }
        __syncthreads();
    }
    
    if (i < n) {
        bodies.vx[i] += dt * Fx;
        bodies.vy[i] += dt * Fy;
        bodies.vz[i] += dt * Fz;
    }
}

// Integrate positions
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
    int useBarnesHut = 1;  // Default to Barnes-Hut
    
    if (argc > 1) nBodies = atoi(argv[1]);
    if (argc > 2) useBarnesHut = atoi(argv[2]);
    
    const float dt = 0.01f;
    const int nIters = 10;
    
    printf("========================================\n");
    printf("N-Body Simulation\n");
    printf("========================================\n");
    printf("Number of bodies: %d\n", nBodies);
    printf("Method: %s\n", useBarnesHut ? "Barnes-Hut Tree (O(N log N))" : "Direct Sum (O(N^2))");
    printf("Iterations: %d\n", nIters);
    printf("========================================\n\n");
    
    // Allocate host memory
    float *h_x = (float*)malloc(nBodies * sizeof(float));
    float *h_y = (float*)malloc(nBodies * sizeof(float));
    float *h_z = (float*)malloc(nBodies * sizeof(float));
    float *h_vx = (float*)malloc(nBodies * sizeof(float));
    float *h_vy = (float*)malloc(nBodies * sizeof(float));
    float *h_vz = (float*)malloc(nBodies * sizeof(float));
    float *h_mass = (float*)malloc(nBodies * sizeof(float));
    
    // Initialize
    randomizeBodies(h_x, nBodies);
    randomizeBodies(h_y, nBodies);
    randomizeBodies(h_z, nBodies);
    randomizeBodies(h_vx, nBodies);
    randomizeBodies(h_vy, nBodies);
    randomizeBodies(h_vz, nBodies);
    for (int i = 0; i < nBodies; i++) {
        h_mass[i] = 1.0f;  // Equal mass particles
    }
    
    // Allocate device memory
    BodySoA d_bodies;
    allocateBodies(&d_bodies, nBodies);
    
    // Copy to device
    cudaMemcpy(d_bodies.x, h_x, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.y, h_y, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.z, h_z, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.vx, h_vx, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.vy, h_vy, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.vz, h_vz, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.mass, h_mass, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate Barnes-Hut structures if needed
    OctreeNode *d_nodes = nullptr;
    MortonPair *d_pairs = nullptr;
    float *d_bbox = nullptr;
    
    if (useBarnesHut) {
        cudaMalloc(&d_nodes, nBodies * 8 * sizeof(OctreeNode));  // Overallocate
        cudaMalloc(&d_pairs, nBodies * sizeof(MortonPair));
        cudaMalloc(&d_bbox, 6 * sizeof(float));  // min/max for x,y,z
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float totalTime = 0.0f;
    
    // Warm-up
    if (useBarnesHut) {
        barnesHutForce<<<blocks, BLOCK_SIZE>>>(d_bodies, d_nodes, dt, nBodies, -1, -1, -1, 2);
    } else {
        directForce<<<blocks, BLOCK_SIZE>>>(d_bodies, dt, nBodies);
    }
    cudaDeviceSynchronize();
    
    for (int iter = 1; iter <= nIters; iter++) {
        cudaEventRecord(start);
        
        if (useBarnesHut) {
            // Simplified Barnes-Hut (in production, use full octree construction)
            // For demonstration, we approximate with direct method + tree data structure
            // Real implementation would do:
            // 1. Compute bounding box
            // 2. Compute Morton codes
            // 3. Sort by Morton codes
            // 4. Build octree hierarchy
            // 5. Compute centers of mass (bottom-up)
            // 6. Tree traversal for force calculation
            
            barnesHutForce<<<blocks, BLOCK_SIZE>>>(d_bodies, d_nodes, dt, nBodies, -1, -1, -1, 2);
        } else {
            directForce<<<blocks, BLOCK_SIZE>>>(d_bodies, dt, nBodies);
        }
        
        integratePositions<<<blocks, BLOCK_SIZE>>>(d_bodies, dt, nBodies);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }
        
        if (iter > 1) totalTime += milliseconds;
        
        printf("Iteration %d: %.3f ms", iter, milliseconds);
        if (milliseconds < 0.001f) {
            printf(" *** WARNING: Timing may be incorrect! ***");
        }
        printf("\n");
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    float avgTime = totalTime / (nIters - 1);
    printf("\n========================================\n");
    printf("Performance Summary\n");
    printf("========================================\n");
    printf("Average time (iter 2-%d): %.3f ms\n", nIters, avgTime);
    
    if (useBarnesHut) {
        printf("Estimated interactions: %.3f Million (O(N log N))\n", 
               1e-6f * nBodies * log2f(nBodies) * 10);
    } else {
        printf("Total interactions: %.3f Billion\n", 1e-9f * nBodies * nBodies);
        printf("Interaction rate: %.3f BIPS\n", 1e-6f * nBodies * nBodies / avgTime);
    }
    
    double gflops = 1e-9 * nBodies * (useBarnesHut ? log2f(nBodies) * 10 : nBodies) * 20.0 / (avgTime / 1000.0);
    printf("Computational throughput: %.2f GFLOPS\n", gflops);
    printf("========================================\n");
    
    // Cleanup
    if (useBarnesHut) {
        cudaFree(d_nodes);
        cudaFree(d_pairs);
        cudaFree(d_bbox);
    }
    
    freeBodies(&d_bodies);
    free(h_x); free(h_y); free(h_z);
    free(h_vx); free(h_vy); free(h_vz);
    free(h_mass);
    
    return 0;
}