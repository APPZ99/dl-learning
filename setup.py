import glob
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name='pytorch_cppcuda_learning',
    version='1.0',
    author='APPZ99',
    author_email='appzz997@gmail.com',
    description='pytorch_cppcuda_learning',
    long_description='cpytorch_cppcuda_learning',
    ext_modules=[
        CUDAExtension(
            name='pytorch_cppcuda_learning',
            sources= sources,
            include_dirs = include_dirs,
            extra_compile_args = {'cxx': ['-O2'],
                                  'nvcc':['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)