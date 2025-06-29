import cupy as cu
import numpy as np

array_cpu = np.random.randint(0, 255, (2000, 2000))

print(f"Array size = {array_cpu.nbytes}")
print(f"Array : {array_cpu}")

