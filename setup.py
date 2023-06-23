# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = "Spyx is a compact library built on top of DeepMind's Haiku library, enabling easy construction of spiking neural network models."


requires = (
    [
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "dm-haiku",
        "numpy",
    ],
)

# This call to setup() does all the work
setup(
    name="spyx",
    version="0.0.6",
    description="Spyx: SNNs in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kmheckel/spyx",
    author="Kade Heckel",
    author_email="example@email.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=["spyx"],
    include_package_data=True,
    install_requires=requires
)
