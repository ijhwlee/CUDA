﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t saxpyWithCuda(float* c, const float* a, const float* b, float m, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    //int i = threadIdx.x;
    int i = blockIdx.x * blockIdx.y + threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void saxpyKernel(float* c, const float* a, const float* b, const float m)
{
  //int i = threadIdx.x;
  int i = blockIdx.x * blockIdx.y + threadIdx.x;
  c[i] = m*a[i] + b[i];
}

float initializeMatrix(float* a, float* b, float a_value, float b_value, float m, int size)
{
  for (int i = 0; i < size; i++)
  {
    a[i] = a_value;
    b[i] = b_value;
  }
  return (m* a_value + b_value);
}

void initializeMatrixInt(int* a, int* b, int size)
{
  for (int i = 0; i < size; i++)
  {
    a[i] = i+1;
    b[i] = 10*(i+1);
  }
}

float checkError(float* c, float *maxError, float check_value, int size)
{
  float errorSum = 0.0f;
  float tmp = 0.0f;
  *maxError = 0.0f;
  for (int i = 0; i < size; i++)
  {
    tmp = abs(c[i] - check_value);
    errorSum += tmp;
    *maxError = (*maxError > tmp ? *maxError : tmp);
  }
  return errorSum;
}


int main()
{
    const int arraySize = 1 << 10;
    int a[arraySize] = { 1 };
    int b[arraySize] = { 10 };
    int c[arraySize] = { 0 };
    float af[arraySize] = { 0 };
    float bf[arraySize] = { 0 };
    float cf[arraySize] = { 0 };

    initializeMatrixInt(a, b, arraySize);
    clock_t begin = clock();
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Times for vector addition of size %d is %.3f\n", arraySize, time_spent);

    float a_value = 2.0f;
    float b_value = 3.0f;
    float m = 2.0f;
    float check_value = 0.0f;
    check_value = initializeMatrix(af, bf, a_value, b_value, m, arraySize);
    // calculate m*a + b vectors in parallel.
    begin = clock();
    for (int i = 0; i < 100; i++)
    {
      printf("Run = %d\n", i);
      cudaStatus = saxpyWithCuda(cf, af, bf, m, arraySize);
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpyWithCuda failed!");
        return 1;
      }

      // cudaDeviceReset must be called before exiting in order for profiling and
      // tracing tools such as Nsight and Visual Profiler to show complete traces.
      cudaStatus = cudaDeviceReset();
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
      }
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Times for vector m*a + b of size %d is %.3f\n", arraySize, time_spent);
    float errorSum = 0.0f;
    float maxError = 0.0f;
    errorSum = checkError(cf, &maxError, check_value, arraySize);
    printf("Error sum = %f, max error = %f\n", errorSum, maxError);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<(size+1023)/1024, 1024 >> >(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t saxpyWithCuda(float* c, const float* a, const float* b, float m, unsigned int size)
{
  float* dev_a = 0;
  float* dev_b = 0;
  float* dev_c = 0;
  cudaError_t cudaStatus;

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  // Allocate GPU buffers for three vectors (two input, one output)    .
  cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

  cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

  // Launch a kernel on the GPU with one thread for each element.
  //saxpyKernel <<<1, size >>> (dev_c, dev_a, dev_b, m);
  //saxpyKernel <<<1, size >>> (dev_c, dev_a, dev_b, m);
  saxpyKernel <<<(size+1023)/1024, 1024 >>> (dev_c, dev_a, dev_b, m);

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "saxpyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    goto Error;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

Error:
  cudaFree(dev_c);
  cudaFree(dev_a);
  cudaFree(dev_b);

  return cudaStatus;
}
