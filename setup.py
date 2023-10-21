from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension



setup(
    name='pytorch_cppcuda_learning',
    version='1.0',
    author='APPZ99',
    author_email='appzz997@gmail.com',
    description='pytorch_cppcuda_learning',
    long_description='cpytorch_cppcuda_learning',
    ext_modules=[
        CppExtension(
            name='pytorch_cppcuda_learning',
            sources=['interpolation.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)