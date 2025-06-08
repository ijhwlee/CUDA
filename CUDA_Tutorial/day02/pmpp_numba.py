import os, time, sys, math, gzip, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from urllib.request import urlretrieve
from pathlib import Path
from numba import cuda
import numpy as np

import torchvision.transforms.functional as tvf
from torchvision import io

# define image save function
def save_image(x, filename, figsize=(4,3), **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape) == 3:
        x = x.permute(1,2,0) # CHW -> HWC
    plt.imshow(x, **kwargs)
    plt.savefig(filename)

# define conversion function in CUDA kernel
@cuda.jit
def rgb2gray_kernel(x, y):
    idx = cuda.grid(1)
    n = len(y)
    if idx < n:
        y[idx] = 0.2989*x[idx] + 0.5860*x[idx+n] + 0.1140*x[idx + 2*n]

def rgb2gray_numba(x):
    c, h, w = x.shape
    n = h*w
    x = x.flatten()
    y = np.zeros(n).astype(np.uint8)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    threads_per_block = 1024
    blocks_per_grid = (n + threads_per_block - 1)//threads_per_block
    rgb2gray_kernel[blocks_per_grid, threads_per_block](d_x, d_y)
    y = d_y.copy_to_host()
    return y.reshape(h, w)

# image url
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg?20140729055059'
# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/500px-Golde33443.jpg'

# import example image
path_img = Path('puppy.jpg')
if not path_img.exists():
    urlretrieve(url, path_img)
img = io.read_image('puppy.jpg')
ch0, h0, w0 = img.shape
print(img.shape)
print(img[:3, :3, :6])

reduced_size = 150
if len(sys.argv) >= 2:
    reduced_size = int(sys.argv[1])
else:
    reduced_size = 150
if reduced_size > h0:
    reduced_size = h0
# resize the image
img2 = tvf.resize(img, reduced_size, antialias=True)
ch, h, w = img2.shape
print(f"Resized image Channel : {ch}, Height : {h}, Width {w}, Bytes : {ch*h*w}")

# save resized image
save_image(img2, 'puppy_resize.png')

# RGB to gray conversion
# actual conversion in basic python
start_time = time.perf_counter()
img_g = rgb2gray_numba(img)
end_time = time.perf_counter()
h, w = img_g.shape
ch = 1
print(f"Converted image Channel : {ch}, Height : {h}, Width {w}, Bytes : {ch*h*w}")
start_time1 = time.perf_counter()
img_g2 = rgb2gray_numba(img2)
end_time1 = time.perf_counter()
h1, w1 = img_g2.shape
ch = 1
print(f"Converted image Channel : {ch}, Height : {h1}, Width {w1}, Bytes : {ch*h1*w1}")

# save converted image
save_image(img_g, 'puppy_grey.png', cmap='gray')
save_image(img_g2, 'puppy_grey2.png', cmap='gray')
dtime = end_time - start_time
if dtime < 0.01:
    dtime *=1000;
    unit = "milli seconds"
else:
    unit = "seconds"
print(f'Elapsed time in CUDA Kernel with size {w}X{h}: {dtime:.3f} {unit}')

dtime = end_time1 - start_time1
if dtime < 0.01:
    dtime *=1000;
    unit = "milli seconds"
else:
    unit = "seconds"
print(f'Elapsed time in CUDA Kernel with size {w1}X{h1}: {dtime:.3f} {unit}')
