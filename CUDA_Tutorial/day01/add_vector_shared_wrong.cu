#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define RANDOM_RANGE 500
#define RADIUS 3
#define N (2048*2048)
#define THREADS_PER_BLOCK 512

__global__ void stencil_1d(int *in, int *out, int n)
{
  __shared__ int temp[THREADS_PER_BLOCK + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  if (gindex >= n)
    return;
  // Read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS)
  {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    if (gindex + THREADS_PER_BLOCK >= n)
      temp[lindex + THREADS_PER_BLOCK] = in[n-1];
    else
      temp[lindex + THREADS_PER_BLOCK] = in[gindex + THREADS_PER_BLOCK];
  }
  // Synchronize (ensure all the data is avaiable)
  __syncthreads();
  // Apply the stencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += temp[lindex + offset];
  // Store the result
  out[gindex] = result;
}

void random_ints(int *a, int n)
{
  for(int i = 0; i<n; i++)
  {
    a[i] = rand()%RANDOM_RANGE;
  }
}

int sum_radius(int *a, int k, int n)
{
  int sum = 0;
  for (int i=-RADIUS; i<=RADIUS; i++)
  {
    if (k+i >=0 && k+i < n)
      sum += a[k+i];
    else if (k+i < 0)
      sum += a[0];
    else if (k+i >= n)
      sum += a[n-1];
  }
  return sum;
}

int main()
{
  int *a, *c;		// host copies of a, c
  int *d_a, *d_c;		// device copies of a, c
  int size = N * sizeof(int);	// array size in bytes
  int k, sum;

  // Intialize random seed
  srand((unsigned int)time(NULL));
  // Allocate space for device copies of a, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_c, size);
  // Allocate space for host copies of a, c and setup input values
  a = (int *)malloc(size); random_ints(a, N);
  c = (int *)malloc(size);
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU with N threads
  stencil_1d<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_c, N);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  // Print the result
  sum = sum_radius(a, 0, N);
  printf("Result: c[%d] is %d, sum = %d\n", 0, c[0], sum);
  k = rand()%N;
  sum = sum_radius(a, k, N);
  printf("Result: c[%d] is %d, sum = %d\n", k, c[k], sum);
  sum = sum_radius(a, N-1, N);
  printf("Result: c[%d] is %d, sum = %d\n", N-1, c[N-1], sum);
  // Cleanup
  free(a); free(c);
  cudaFree(d_a); cudaFree(d_c);
  return 0;
}

