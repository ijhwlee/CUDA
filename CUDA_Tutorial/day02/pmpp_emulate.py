import torch, os, time, sys, math, gzip, pickle
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from pathlib import Path

from torch import tensor
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
from torch.utils.cpp_extension import load_inline

# define image save function
def save_image(x, filename, figsize=(4,3), **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape) == 3:
        x = x.permute(1,2,0) # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)
    plt.savefig(filename)

# define conversion function in basic python with kernel emulation
def run_kernel(f, times, *args):
    for i in range(times):
        f(i, *args)

def rgb2grey_k(i, x, out, n):
    out[i] = 0.2989*x[i] + 0.5860*x[i+n] + 0.1140*x[i + 2*n]

def rgb2grey_pyk(x):
    c,h,w = x.shape
    n = h*w
    x = x.flatten()
    res = torch.empty(n, dtype=x.dtype, device=x.device)
    run_kernel(rgb2grey_k, h*w, x, res, n)
    return res.view(h,w)

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
img_g = rgb2grey_pyk(img2)
end_time = time.perf_counter()
h, w = img_g.shape
ch = 1
print(f"Converted image Channel : {ch}, Height : {h}, Width {w}, Bytes : {ch*h*w}")

# save converted image
save_image(img_g, 'puppy_grey.png', cmap='gray')
print(f'Elapsed time in Emulating Kernel with size {w}X{h}: {end_time - start_time:.3f} seconds')

