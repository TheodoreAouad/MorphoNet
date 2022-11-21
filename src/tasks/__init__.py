"""Init package."""

import os


from misc.visualizer import VisualizerCallback
from misc.context import OutputManagment

output_managment = OutputManagment()
output_managment.set()

# Same device ordering as `nvidia-smi`
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
