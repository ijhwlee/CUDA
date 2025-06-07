#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define RANDOM_RANGE 500

__global__ void add(int *a, int *b, int *c)
{
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *a, int n)
{
  for(int i = 0; i<n; i++)
  {
    a[i] = rand()%RANDOM_RANGE;
  }
}

#define N 512

int main()
{
  int *a, *b, *c;		// host copies of a, b, c
  int *d_a, *d_b, *d_c;		// device copies of a, b, c
  int size = N * sizeof(int);	// array size in bytes
  int k;

  // Intialize random seed
  srand((unsigned int)time(NULL));
  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  // Allocate space for host copies of a, b, c and setup input values
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size);
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU with N threads
  add<<<1,N>>>(d_a, d_b, d_c);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  // Print the result
  k = rand()%N;
  printf("Result: c[%d] is %d, a[%d] = %d, b[%d] = %d\n", k, c[k], k, a[k], k, b[k]);
  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}

