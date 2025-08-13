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

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1)/b;}
'''

cuda_src = cuda_begin + r'''
__global__ void rgb_to_grayscale_kernel(unsigned char *x, unsigned char *out, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = 0.2989*x[i] + 0.5860*x[i+n] + 0.1140*x[i + 2*n];
  }
}

torch::Tensor rgb_to_grayscale(torch::Tensor input)
{
  CHECK_INPUT(input);
  int h = input.size(1);
  int w = input.size(2);
  printf("h*w : %d X %d\n", h, w);
  auto output = torch::empty({h,w}, input.options());
  int threads = 512;
  rgb_to_grayscale_kernel<<<cdiv(h*w, threads), threads>>>(
    input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), h*w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
'''

cpp_src = "torch::Tensor rgb_to_grayscale(torch::Tensor input);"

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

# using CUDA 
module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], verbose=True)
print(dir(module))

imgc = img.contiguous().cuda()
imgc2 = img2.contiguous().cuda()

# RGB to gray conversion
# actual conversion in basic python
start_time = time.perf_counter()
img_g = module.rgb_to_grayscale(imgc).cpu()
end_time = time.perf_counter()
h, w = img_g.shape
ch = 1
print(f"Converted image Channel : {ch}, Height : {h}, Width {w}, Bytes : {ch*h*w}")
start_time1 = time.perf_counter()
img_g2 = module.rgb_to_grayscale(imgc2).cpu()
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

