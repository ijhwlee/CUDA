import random
import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Number of CUDA devices
num_device = torch.cuda.device_count()
print("CUDA Device Count:", num_device)

set_device = random.randint(0, num_device-1)
torch.cuda.set_device(set_device)

# Current CUDA device index
print("Current Device Index:", torch.cuda.current_device())

# Name of the current CUDA device
print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# CUDA version PyTorch was compiled with
print("PyTorch CUDA Version:", torch.version.cuda)

