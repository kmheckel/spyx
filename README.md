🏗️ 🚧 This library is under construction!🚧
===========================================

Welcome to Spyx, a library for Spiking Neural Networks in JAX!

![README Art](spyx.png "Title")

Spyx is a compact library built on top of DeepMind's Haiku library, enabling easy construction of spiking neural network models. 

The goal of Spyx is to provide similar capabilities as SNNTorch for the JAX ecosystem, opening up the possibility to incorporate SNNs into a number of GPU-accelerated reinforcement learning environments. Additionally, JAX has become home to several libraries for neuroevolution, and the aim is for Spyx to provide a common framework to compare modern neuroevolution algorithms with surrogate gradient and ANN2SNN conversion techniques.

In the future the aim for Spyx is to include more detailed neuron models such as Hodgkin-Huxley and to incorporate support for executing models trained in Spyx on neuromorphic hardware such as Intel's Loihi architecture.

Installation
============

This library is developed on Ubuntu 22.04 wwith the LambdaLabs stack for handling deep learning frameworks.

As with other libraries built on top of JAX, you need to install jax with GPU if you want to get the full benefit of this library.

Directions for installing JAX with GPU support can be found at the following: https://github.com/google/jax#installation
