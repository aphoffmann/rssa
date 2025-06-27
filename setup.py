from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        'py_rssa._fast_hankel',
        ['py_rssa/py_rssa/_fast_hankel.c'],
        include_dirs=[numpy.get_include()],
        libraries=['fftw3'],
    )
]

setup(
    name='py_rssa',
    version='0.1.0',
    packages=['py_rssa'],
    ext_modules=ext_modules,
)
