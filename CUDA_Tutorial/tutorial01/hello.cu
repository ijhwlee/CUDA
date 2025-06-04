#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_hello()
{
  printf("Hello World from GPU!\n");
}

int main()
{
  printf("Calling cuda_helloEnd of mainEnd of mainEnd of main...\n");
  cuda_hello<<<1,1>>>();
  printf("End of main.\n");
  return 0;
}
