# import torch
# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# for i in range(torch.cuda.device_count()):
#     print((i, torch.cuda.get_device_name(i)))

import torchvision

torchvision.datasets.MNIST("mnist", download=True)
torchvision.datasets.MNIST("mnist", download=True, train=False)