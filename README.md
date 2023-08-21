⚡🧠💻 Welcome to Spyx! 💻🧠⚡
============================
[![DOI](https://zenodo.org/badge/656877506.svg)](https://zenodo.org/badge/latestdoi/656877506) [![PyPI version](https://badge.fury.io/py/spyx.svg)](https://badge.fury.io/py/spyx)
![README Art](spyx.png "Spyx")

Spyx is a compact spiking neural network library built on top of DeepMind's Haiku library.

The goal of Spyx is to provide similar capabilities as SNNTorch for the JAX ecosystem, opening up the possibility to incorporate SNNs into a number of GPU-accelerated reinforcement learning environments. Additionally, JAX has become home to several libraries for neuroevolution, and the aim is for Spyx to provide a common framework to compare modern neuroevolution algorithms with surrogate gradient and ANN2SNN conversion techniques.

The future aim for Spyx is to include tools for building and training spiking phasor networks and building an interface for exporting models to the emerging Neuromorphic Intermediate Representation for deployment on efficient hardware.

Installation:
=============

As with other libraries built on top of JAX, you need to install jax with GPU if you want to get the full benefit of this library. Directions for installing JAX with GPU support can be found at the following: https://github.com/google/jax#installation

Additionally, the data loading is dependent on Tonic, a library for neuromorphic datasets. You will have to install it seperately to avoid creating headaches between dependencies for JAX and PyTorch. 

 https://tonic.readthedocs.io/en/latest/getting_started/install.html

Hardware Requirements:
======================

Spyx achieves extremely high performance by maintaining the entire dataset in the GPU's vRAM; as such a decent amount of memory for both the CPU and GPU are needed to handle the dataset loading and then training. For smaller networks of only several hundred thousand parameters, the training process can be comfortably executed on even laptop GPU's with only 6GB of vRAM. For large SNNs or for neuroevolution it is recommended to use a higher memory card.

Since Spyx is developed on top of the current JAX version, it does not work on Google Colab's TPUs which use an older version. Cloud TPU support will be tested in the near future. Support for GraphCore's IPU's could be possible based on their fork of JAX but has not been explored. 

Why use Spyx?
=============

Other frameworks such as SNNTorch and Norse offer a nice range of features such as training with adjoint gradients or support for IPUs in addition to their wonderful tutorials. Spyx is designed to maximize performance by achieving maximum GPU utilizattion, allowing the training of networks for hundreds of epochs at incredible speed.


Citation:
=========

For now, use the Bibtex entry below or click the badge above the title image to get other formats from Zenodo.


```
@software{kade_heckel_2023_8241588,
  author       = {Kade Heckel},
  title        = {kmheckel/spyx: v0.1.0-beta},
  month        = aug,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {beta},
  doi          = {10.5281/zenodo.8241588},
  url          = {https://doi.org/10.5281/zenodo.8241588}
}
```