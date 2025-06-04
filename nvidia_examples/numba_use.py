import numpy as np
from numba import cuda
import time
import matplotlib.pyplot as plt

@cuda.jit
def add_arrays(a, b, c):
    idx = cuda.grid(1) # Get the thread index
    if idx < c.size:
       c[idx] = a[idx] + b[idx]

def add_arrays_normal(a, b, c):
    for idx in range(c.size):
       c[idx] = a[idx] + b[idx]

def wall_time(N):
    # Host arrays
    # N = 100000000
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)

    start_cuda = time.time()
    # Device arrays
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(d_a)

    # Kernel launch
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    add_arrays[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy results back to the host
    c = d_c.copy_to_host()

    print(c[:10])
    time_cuda = time.time() - start_cuda
    print(f"Array add of size {N} using Cuda is {time_cuda} seconds")

    start_normal = time.time()
    add_arrays_normal(a, b, c)
    print(c[:10])
    time_normal = time.time() - start_normal
    print(f"Array add of size {N} using for loop is {time_normal} seconds")
    return time_cuda, time_normal

array_size = []
size = 1
for idx in range(9):
    size = 10*size
    array_size.append(size)
times_cuda = []
times_normal = []
for idx in range(len(array_size)):
    print("======================================================================")
    print(f"=      idx = {idx}")
    print("======================================================================")
    t1, t2 = wall_time(array_size[idx])
    times_cuda.append(t1)
    times_normal.append(t2)

plt.figure(figsize=(8,6))
plt.plot(array_size, times_cuda, label='CUDA')
plt.plot(array_size, times_normal, label='Array')
plt.legend()
plt.xlabel('Array Size')
plt.ylabel('Walltime')
plt.title('Walltime Comparison')
plt.savefig('wall_time.png')

plt.figure(figsize=(8,6))
plt.loglog(array_size, times_cuda, label='CUDA')
plt.loglog(array_size, times_normal, label='Array')
plt.legend()
plt.xlabel('Array Size')
plt.ylabel('Walltime')
plt.title('Walltime Comparison')
plt.savefig('wall_time_log.png')

plt.figure(figsize=(8,6))
plt.loglog(array_size, np.array(times_normal)/np.array(times_cuda))
plt.xlabel('Array Size')
plt.ylabel('Walltime Ratio')
plt.title('Walltime Comparison')
plt.savefig('wall_time_ratio.png')
