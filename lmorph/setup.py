from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torch_lmorph",
    version="0.0.0",
    description="PyTorch implementation of the LehmerMorph operation.",
    author="Alexandre Kirszenberg",
    author_email="alexandre.kirszenberg@lrde.epita.fr",
    packages=find_packages(exclude=("test",)),
    ext_modules=[
        CUDAExtension(
            name="lmorph._C",
            sources=[
                "lmorph/csrc/ValidDim.cpp",
                "lmorph/csrc/LMorph.cpp",
                "lmorph/csrc/LMorph_cuda.cu",
            ],
            include_dirs=["lmorph/csrc"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
