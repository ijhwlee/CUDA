import os, time, sys, math, gzip, pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

RANDOM_RANGE = 500

# define image save function
def print_elapsed(elapsed):
    if elapsed < 1.0e-5:
        print(f"Elapsed time : {elapsed*1.0e6:.3f} micro seconds.")
    elif elapsed < 1.0e-2:
        print(f"Elapsed time : {elapsed*1.0e3:.3f} milli seconds.")
    else:
        print(f"Elapsed time : {elapsed:.3f} seconds.")

def random_ints(x, n):
    x = np.random.randint(0, RANDOM_RANGE, n)
    return x

# define conversion function in CUDA kernel
def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
            extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define THREADS_PER_BLOCK 512
inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1)/b;}
'''

cuda_src = cuda_begin + r'''
__global__ void add_kernel(unsigned int *x, unsigned int *y, unsigned int *z, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    z[i] = x[i] + y[i];
  }
}

torch::Tensor add_vector_torch(torch::Tensor x, torch::Tensor y, torch::Tensor z)
{
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  int n = x.size(1);
  printf(" n = %d\n", n);
  int threads = THREADS_PER_BLOCK;
  add_kernel<<<cdiv(n, threads), threads>>>(x.data_ptr<unsigned int>(), y.data_ptr<unsigned int>(), z.data_ptr<unsigned int>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return z;
}
'''

cpp_src = "torch::Tensor add_vector_torch(torch::Tensor x, torch::Tensor y, torch::Tensor z);"

N = 2048 * 2048
THREADS_PER_BLOCK = 512

x = np.zeros(N).astype(np.uint32)
y = np.zeros(N).astype(np.uint32)
z = np.zeros(N).astype(np.uint32)
k = np.random.randint(0, N)
x = random_ints(x, N)
y = random_ints(y, N)
x_t = torch.from_numpy(np.array([x], dtype=np.uint32))
y_t = torch.from_numpy(np.array([y], dtype=np.uint32))
z_t = torch.from_numpy(np.array([z], dtype=np.uint32))
print(x_t.shape)
print(x_t.flatten().shape)

module = load_cuda(cuda_src, cpp_src, ['add_vector_torch'], verbose=True)
print(dir(module))
start = time.perf_counter()
x_cuda = x_t.contiguous().cuda()
y_cuda = y_t.contiguous().cuda()
z_cuda = z_t.contiguous().cuda()
print(x_cuda.shape)
print(x_cuda.flatten().shape)
blocks_per_grid = (N + THREADS_PER_BLOCK - 1)//THREADS_PER_BLOCK
z1 = module.add_vector_torch(x_cuda, y_cuda, z_cuda).cpu()
z = z1.numpy()[0]
end = time.perf_counter()

print(f"Result: Size = {N}, x[{k}] = {x[k]}, y[{k}] = {y[k]}, z[{k}] = {z[k]}")
print_elapsed(end - start)

