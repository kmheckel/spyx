.. Spyx documentation master file, created by
   sphinx-quickstart on Thu Jun 22 10:06:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Spyx's documentation!
================================

Spyx is a compact spiking neural network library built on top of DeepMind's Haiku package.

Spyx promises the flexibility and extensibility offered by PyTorch-based SNN libraries while enabling extremely efficient training on high-performance hardware at speeds comparable to or faster than SNN frameworks that have custom CUDA implementataions.


Be sure to go give it a star on Github: https://github.com/kmheckel/spyx

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   introduction
   quickstart
   examples/surrogate_gradient/SurrogateGradientTutorial
   examples/neuroevolution/cartpole_evo

   examples/surrogate_gradient/shd_sg_neuron_model_comparison
   examples/surrogate_gradient/shd_sg_surrogate_comparison
   examples/surrogate_gradient/shd_sg_template
   examples/surrogate_gradient/shd_eprop

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
