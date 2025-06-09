#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define RANDOM_RANGE 500

__global__ void add(int *a, int *b, int *c, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] + b[index];
}

void random_ints(int *a, int n)
{
  for(int i = 0; i<n; i++)
  {
    a[i] = rand()%RANDOM_RANGE;
  }
}

void print_elapsed(double elapsed_time)
{
  if (elapsed_time < 1.0e-5)
  {
    printf("Elapsed time : %.3f micro seconds.\n", elapsed_time*1.0e6);
  }
  else if (elapsed_time < 1.0e-2)
  {
    printf("Elapsed time : %.3f milli seconds.\n", elapsed_time*1.0e3);
  }
  else
  {
    printf("Elapsed time : %.3f seconds.\n", elapsed_time);
  }
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

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
  // Launch add() kernel on GPU with N threads
  add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  end = clock();
  // Print the result
  k = rand()%N;
  printf("Result: Size = %d, c[%d] = %d, a[%d] = %d, b[%d] = %d\n", N, k, c[k], k, a[k], k, b[k]);
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  print_elapsed(elapsed_time);
  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}

