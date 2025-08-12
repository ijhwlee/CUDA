#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define RANDOM_RANGE 500

__global__ void add(int *a, int *b, int *c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

void random_ints(int *a, int n)
{
  for(int i = 0; i<n; i++)
  {
    a[i] = rand()%RANDOM_RANGE;
  }
}

void print_elapsed(double elapsed_time, const char * msg)
{
  if (elapsed_time < 1.0e-5)
  {
    printf("Elapsed time for %s : %.3f micro seconds.\n", msg, elapsed_time*1.0e6);
  }
  else if (elapsed_time < 1.0e-2)
  {
    printf("Elapsed time for %s : %.3f milli seconds.\n", msg, elapsed_time*1.0e3);
  }
  else
  {
    printf("Elapsed time for %s : %.3f seconds.\n", msg, elapsed_time);
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
  clock_t start, end, start_mem, end_mem;
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
  end_mem = clock();
  // Launch add() kernel on GPU with N threads
  add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
  // Copy result back to host
  start_mem = clock();
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  end = clock();
  // Print the result
  k = rand()%N;
  printf("Result: c[%d] is %d, a[%d] = %d, b[%d] = %d\n", k, c[k], k, a[k], k, b[k]);
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  print_elapsed(elapsed_time, "Device add with memory copy");
  elapsed_time = ((double)(end_mem - start)) / CLOCKS_PER_SEC;
  print_elapsed(elapsed_time, "Memory Copy to Device");
  elapsed_time = ((double)(end - start_mem)) / CLOCKS_PER_SEC;
  print_elapsed(elapsed_time, "Memory Copy to Host");
  elapsed_time = ((double)(start_mem - end_mem)) / CLOCKS_PER_SEC;
  print_elapsed(elapsed_time, "Device add operation");
  start = clock();
  for(int idx = 0; idx < N; idx++)
  {
    c[idx] = a[idx] + b[idx];
  }
  end = clock();
  k = rand()%N;
  printf("Result: c[%d] is %d, a[%d] = %d, b[%d] = %d\n", k, c[k], k, a[k], k, b[k]);
  elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  print_elapsed(elapsed_time, "C for loop");
  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}

