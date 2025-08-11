#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define RANDOM_RANGE 500

__global__ void add(int *a, int *b, int *c)
{
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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
  clock_t start, end;
  double elapsed_time;

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
  start = clock();
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU
  add<<<N,1>>>(d_a, d_b, d_c);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  end = clock();
  // Print the result
  k = rand()%N;
  printf("Result: c[%d] is %d, a[%d] = %d, b[%d] = %d\n", k, c[k], k, a[k], k, b[k]);
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  if (elapsed_time < 0.01)
  {
    printf("Elapsed time : %.3f milli seconds.\n", elapsed_time*1000);
  }
  else
  {
    printf("Elapsed time : %.3f seconds.\n", elapsed_time);
  }
  start = clock();
  for(int idx = 0; idx < N; idx++)
  {
    c[idx] = a[idx] + b[idx];
  }
  end = clock();
  k = rand()%N;
  printf("Result: c[%d] is %d, a[%d] = %d, b[%d] = %d\n", k, c[k], k, a[k], k, b[k]);
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  if (elapsed_time < 0.01)
  {
    printf("Elapsed time : %.3f milli seconds.\n", elapsed_time*1000);
  }
  else
  {
    printf("Elapsed time : %.3f seconds.\n", elapsed_time);
  }
  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}

