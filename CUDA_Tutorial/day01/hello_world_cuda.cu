#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mykernel(void)
{
  printf("Hello World from Device code!\n");
}

int main()
{
  mykernel<<<1,1>>>();
  printf("Hello World!\n");
  return 0;
}

