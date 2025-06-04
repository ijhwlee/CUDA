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

int main(int argc, char * argv[])
{
  float *a, *b, *out;
  float *d_a, *d_b, *d_out;
  int num_thread;

  if(argc == 2)
  {
    num_thread = atoi(argv[1]);
  }
  else
  {
    num_thread = 1;
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
  printf("Calling vector_add with thread number %d...\n", num_thread);
  vector_add<<<1,num_thread>>>(d_out, d_a, d_b, N);
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
