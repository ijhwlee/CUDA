import os, time, sys, math, gzip, pickle
import matplotlib.pyplot as plt
from numba import cuda
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
@cuda.jit
def add(x, y, z, n):
    idx = cuda.grid(1)
    n = len(y)
    if idx < n:
        z[idx] = x[idx] + y[idx]

N = 2048 * 2048
THREADS_PER_BLOCK = 512

x = np.zeros(N).astype(np.uint32)
y = np.zeros(N).astype(np.uint32)
z = np.zeros(N).astype(np.uint32)
k = np.random.randint(0, N)
x = random_ints(x, N)
y = random_ints(y, N)

start = time.perf_counter()
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_z = cuda.to_device(z)
blocks_per_grid = (N + THREADS_PER_BLOCK - 1)//THREADS_PER_BLOCK
add[blocks_per_grid, THREADS_PER_BLOCK](d_x, d_y, d_z, N)
z = d_z.copy_to_host()
end = time.perf_counter()

print(f"Result: Size = {N}, x[{k}] = {x[k]}, y[{k}] = {y[k]}, z[{k}] = {z[k]}")
print_elapsed(end - start)

