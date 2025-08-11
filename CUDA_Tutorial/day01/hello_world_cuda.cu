#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mykernel(void)
{
  printf("Hello World from Device code! blockIdx.x = %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
  mykernel<<<5,5>>>();
  cudaDeviceSynchronize();  // Ensure all output is flushed
  printf("Hello World!\n");
  return 0;
}

