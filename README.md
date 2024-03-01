⚡🧠💻 Welcome to Spyx! 💻🧠⚡
============================
[![arXiv](https://img.shields.io/badge/arXiv-2402.18994-b31b1b.svg)](https://arxiv.org/abs/2402.18994) [![DOI](https://zenodo.org/badge/656877506.svg)](https://zenodo.org/badge/latestdoi/656877506) [![PyPI version](https://badge.fury.io/py/spyx.svg)](https://badge.fury.io/py/spyx)

[![](https://dcbadge.vercel.app/api/server/TCYQFWsBwj)](https://discord.gg/TCYQFWsBwj)


![README Art](spyx.png "Spyx")

Why use Spyx?
=============

Spyx (pronounced "spikes") is a compact spiking neural network library built on top of DeepMind's Haiku package, offering the flexibility and extensibility of PyTorch-based frameworks while enabling the extreme perfomance of SNN libraries which implement custom CUDA kernels for their dynamics. 

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


Research and Projects Using Spyx:
=================================

Experiments/Benchmarks used in the Spyx Paper: [Benchmark Notebooks](https://github.com/kmheckel/spyx/tree/main/research/paper)

Master's Thesis: Neuroevolution of Spiking Neural Networks [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10620442.svg)](https://doi.org/10.5281/zenodo.10620442)

*** Your projects and research could be here! ***


Contributing:
=============

If you'd like to contribute, head on over to the issues page to find proposed enhancements and leave a comment! Also head over to the Open Neuromorphic Discord server to ask questions!

Citation:
=========

If you find Spyx useful in your work please cite it using the following Bibtex entries:

```
@misc{heckel2024spyx,
    title={Spyx: A Library for Just-In-Time Compiled Optimization of Spiking Neural Networks},
    author={Kade M. Heckel and Thomas Nowotny},
    year={2024},
    eprint={2402.18994},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

```
@software{kade_heckel_2024_10635178,
  author       = {Kade Heckel and
                  Steven Abreu and
                  Gregor Lenz and
                  Thomas Nowotny},
  title        = {kmheckel/spyx: v0.1.17},
  month        = feb,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {camera-ready},
  doi          = {10.5281/zenodo.10635178},
  url          = {https://doi.org/10.5281/zenodo.10635178}
}
```
