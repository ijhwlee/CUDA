import cupy as cu
import numpy as np
import time, sys

def print_elapsed(elapsed):
    if elapsed < 1.0e-5:
        print(f"Elapsed time : {elapsed*1.0e6:.3f} micro seconds.")
    elif elapsed < 1.0e-2:
        print(f"Elapsed time : {elapsed*1.0e3:.3f} milli seconds.")
    else:
        print(f"Elapsed time : {elapsed:.3f} seconds.")

def check_result(a, b):
    diff = np.square(a) - b
    diff = np.sum(diff)
    return diff

array_cpu = np.random.randint(0, 255, (2000, 2000))
array_device = cu.array(array_cpu)

print(f"Array size = {array_cpu.nbytes}")
print(f"Array : {array_cpu}")

squared = cu.ElementwiseKernel(
   'int64 x',
   'float64 z',
   'z = x * x',
   'squared')
start = time.perf_counter()
z = squared(array_device)
end = time.perf_counter()
print("Elapsed time for square")
print_elapsed(end - start)

#print(f"a = {array_cpu}")
#print(f"z = {z}")
start = time.perf_counter()
print(f"Check result = {check_result(array_device, z)}")
end = time.perf_counter()
print("Elapsed time for check")
print_elapsed(end - start)
