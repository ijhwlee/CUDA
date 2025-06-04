#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1.0e-6

__global__ void vector_add(float *out, float *a, float *b, int n)
{
  //printf("Hello World from thread %d vector_add in GPU!\n", threadIdx.x);
  int index = threadIdx.x;
  int stride = blockDim.x;
  for(int i = index; i<n; i+=stride)
  {
    out[i] = a[i] + b[i];
  }
}

void initialize_array(float *a, float value, int n)
{
  for(int i=0; i<n; i++)
  {
    a[i] = value;
  }
}

void show_device_property(cudaDeviceProp *prop, int i)
{
  printf("====================================================================\n");
  printf("       Device property for Device number %d\n", i);
  printf("====================================================================\n");
  printf(" Name \t\t\t: %s\n", prop->name);
  printf(" Memory \t\t: %ld\n", prop->totalGlobalMem);
  printf(" Max. Thread per Block \t: %d\n", prop->maxThreadsPerBlock);
  printf(" Max. Thread Dim \t: %d, %d, %d\n", prop->maxThreadsDim[0],prop->maxThreadsDim[1],prop->maxThreadsDim[2]);
  printf(" Max. Grid Size \t: %d, %d, %d\n", prop->maxGridSize[0],prop->maxGridSize[1],prop->maxGridSize[2]);
  printf("\n\n");
}

int main(int argc, char * argv[])
{
  float *a, *b, *out;
  float *d_a, *d_b, *d_out;
  int block_size, grid_size;
  int num_devices;
  int maxThreadSize;

  maxThreadSize = pow(2,31)-1; // arbitrary big number
  cudaGetDeviceCount(&num_devices);
  printf("Number of Devicdes %d\n", num_devices);
  for(int i=0; i<num_devices; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    show_device_property(&prop, i);
    if(prop.maxThreadsPerBlock < maxThreadSize)
    {
      maxThreadSize = prop.maxThreadsPerBlock;
    }
  }

  if(argc >= 2)
  {
    block_size = atoi(argv[1]);
  }
  else
  {
    block_size = 1;
  }
  if(block_size > maxThreadSize)
  {
    printf("Warning requested block size %d is larger than maximum value %d, set to maximum value.\n", block_size, maxThreadSize);
    block_size = maxThreadSize;
  }
  if(argc >= 3)
  {
    grid_size = atoi(argv[2]);
  }
  else
  {
    grid_size = ((N + block_size -1 ) / block_size);
  }
  
  a = (float *)malloc(sizeof(float) * N);
  b = (float *)malloc(sizeof(float) * N);
  out = (float *)malloc(sizeof(float) * N);
  initialize_array(a, 1.0f, N);
  initialize_array(b, 2.0f, N);

  cudaMalloc((void **)&d_a, sizeof(float) * N);
  cudaMalloc((void **)&d_b, sizeof(float) * N);
  cudaMalloc((void **)&d_out, sizeof(float) * N);
  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
  printf("Calling vector_add with grid_size %d, block_size %d...\n", grid_size, block_size);
  vector_add<<<grid_size, block_size>>>(d_out, d_a, d_b, N);
  printf("Returned from vector_add.\n");
  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  for(int i=0; i<N; i++)
  {
    assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
  }
  printf("out[0] = %f\n", out[0]);

  cudaFree(d_a);cudaFree(d_b);cudaFree(d_out);
  free(a);free(b);free(out);

  printf("PASSED! End of main.\n");
  return 0;
}
