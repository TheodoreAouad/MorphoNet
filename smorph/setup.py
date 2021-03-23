from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torch_smorph",
    version="0.0.0",
    description="PyTorch implementation of the SmoothMorph operation.",
    author="Alexandre Kirszenberg",
    author_email="alexandre.kirszenberg@lrde.epita.fr",
    packages=find_packages(exclude=("test",)),
    ext_modules=[
        CUDAExtension(
            name="smorph._C",
            sources=[
                "smorph/csrc/ValidDim.cpp",
                "smorph/csrc/SMorph.cpp",
                "smorph/csrc/SMorph_cuda.cu",
            ],
            include_dirs=["smorph/csrc"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
