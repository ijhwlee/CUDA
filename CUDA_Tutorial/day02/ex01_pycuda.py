# import and initialize pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time, sys

# define image save function
def print_elapsed(elapsed):
    if elapsed < 1.0e-5:
        print(f"Elapsed time : {elapsed*1.0e6:.3f} micro seconds.")
    elif elapsed < 1.0e-2:
        print(f"Elapsed time : {elapsed*1.0e3:.3f} milli seconds.")
    else:
        print(f"Elapsed time : {elapsed:.3f} seconds.")

N = 16
BLOCK_SIZE = 1

if len(sys.argv) == 3:
    N = int(sys.argv[1])
    BLOCK_SIZE = int(sys.argv[2])

# Create a matrix of random numbers and set it to be 32-bits float
a = np.random.randn(N)
a = a.astype(np.float32)

# Allocate space on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
N_gpu = np.int32(N)

# Transfer the data to the GPU
cuda.memcpy_htod(a_gpu, a)

# Write our GPU kernel inside of our script
module = SourceModule("""
        __global__ void double_array(float *a, int n)
        {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if(idx < n)
            a[idx] *= 2;
        }
        """)

# Launch the kernel
start = time.perf_counter()
grid_size = (N + BLOCK_SIZE - 1)//BLOCK_SIZE
function = module.get_function("double_array")
function(a_gpu, N_gpu, block=(BLOCK_SIZE, 1, 1), grid=(grid_size, 1, 1))

# Place holder for the result
a_doubled = np.empty_like(a)

# Copy the result back from the GPU
cuda.memcpy_dtoh(a_doubled, a_gpu)
end = time.perf_counter()

# Print the original and result
print(f"Original : {a}")
print(f"Doubled : {a_doubled}")
print(f"Size = {N}")
print_elapsed(end - start)

