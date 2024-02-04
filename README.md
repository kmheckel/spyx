⚡🧠💻 Welcome to Spyx! 💻🧠⚡
============================
[![DOI](https://zenodo.org/badge/656877506.svg)](https://zenodo.org/badge/latestdoi/656877506) [![PyPI version](https://badge.fury.io/py/spyx.svg)](https://badge.fury.io/py/spyx)

[![](https://dcbadge.vercel.app/api/server/TCYQFWsBwj)](https://discord.gg/TCYQFWsBwj)


![README Art](spyx.png "Spyx")

Why use Spyx?
=============

Spyx is a compact spiking neural network library built on top of DeepMind's Haiku package, offering the flexibility and extensibility of PyTorch-based frameworks while enabling the extreme perfomance of SNN libraries which implement custom CUDA kernels for their dynamics. 

The library currently supports training SNNs via surrogate gradient descent and neuroevolution, with additional capabilities such as ANN2SNN conversion and Phasor Networks being planned for the future. Spyx offers a number of predefined neuron models but is designed for it to be easy to define your own and plug it into a model; the hope is to soon include definitions of SpikingRWKV and other more sophisticated model blocks into the framework.

Installation:
=============

As with other libraries built on top of JAX, you need to install jax with GPU if you want to get the full benefit of this library. Directions for installing JAX with GPU support can be found at the following: https://github.com/google/jax#installation

The best way to install and run Spyx is if you install it into a container/environment that already has JAX and PyTorch installed.

The spyx.data submodule contains some pre-built dataloaders for use with spyx - to install the depedencies for it run the command `pip install spyx[data]`

Hardware Requirements:
======================

Spyx achieves extremely high performance by maintaining the entire dataset in the GPU's vRAM; as such a decent amount of memory for both the CPU and GPU are needed to handle the dataset loading and then training. For smaller networks of only several hundred thousand parameters, the training process can be comfortably executed on even laptop GPU's with only 6GB of vRAM. For large SNNs or for neuroevolution it is recommended to use a higher memory card.

Since Spyx is developed on top of the current JAX version, it does not work on Google Colab's TPUs which use an older version. Cloud TPU support will be tested in the near future.


Contributing:
=============

If you'd like to contribute, head on over to the issues page to find proposed enhancements and leave a comment! Also head over to the Open Neuromorphic Discord server to ask questions!

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
