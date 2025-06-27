from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        'pyrssa._fast_hankel',
        ['pyrssa/_fast_hankel.c'],
        include_dirs=[numpy.get_include()],
        libraries=['fftw3'],
    )
]

setup(
    name='pyrssa',
    version='0.1.0',
    packages=['pyrssa'],
    ext_modules=ext_modules,
)
