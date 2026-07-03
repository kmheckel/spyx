# Spyx Modernization & Upgrade Plan

This plan outlines the strategic steps to upgrade `Spyx` from a compact, Haiku-based SNN library into a modern, feature-rich framework optimized for spiking and recurrent sequence models.

## Goal
To evolve `Spyx` into the fastest and most flexible JAX-based library for spiking neural networks and quantized recurrent models (SSMs/H-Nets) by adopting **Flax NNX** and supporting advanced training avenues like Phasor Networks and Neuroevolution.

## Proposed Changes

### 1. Framework Transition: From Haiku to Flax NNX
Flax NNX provides a modern, module-based API that simplifies state management while retaining the full power of JAX. It is the ideal foundation for the next generation of Spyx.

*   **[MODIFY] Core Architecture**: Refactor `spyx.layers` to use `nnx.Module`.
*   **[BENEFIT]**: Native state handling and the "New Flax" experience make the library more maintainable and user-friendly.

### 2. Recurrent Sequence Models: SSMs & H-Nets
Extend Spyx beyond simple SNNs to support state-of-the-art recurrent sequence models, with a focus on efficient JAX implementations.

*   **[NEW] State Space Models (SSMs)**: Implement optimized S4/Mamba-style layers within the Spyx ecosystem.
*   **[NEW] H-Nets**: Implement Hierarchical Networks for long-range temporal dependencies.
*   **[BENEFIT]**: Broadens Spyx's applicability to general time-series and sequence processing tasks.

### 3. Quantization & Efficiency
Position Spyx as a leader in training high-performance quantized neural networks.

*   **[NEW] `spyx.quantization`**: Tools for training with low-precision weights and activations, specifically tailored for SSMs and SNNs.
*   **[BENEFIT]**: Enables deployment on resource-constrained neuromorphic and edge hardware.

### 4. Focused Training Avenues
Streamline the library's training mechanisms around three core pillars:

*   **BPTT (Backpropagation Through Time)**: Standard, high-performance gradient-based training using surrogate gradients.
*   **Neuroevolution**: Scaling population-based training (CMA-ES, PGPE) via JAX's `vmap`, leveraging the **evosax** library for robust, hardware-accelerated evolutionary strategies.
*   **Phasor Networks**: Implementing `spyx.phasor` for fast, conversion-less training of temporal models.

## Verification Plan

### Automated Tests
*   **Comparative Benchmarks**: Run performance tests against standard JAX implementations of SSMs and other SNN libraries.
*   **Quantization Robustness**: Verify that quantized SSMs/H-Nets maintain accuracy compared to full-precision counterparts.

### Manual Verification
*   **Tutorial Series**: Create new Flax NNX-based tutorials for:
    1. Training SNNs with BPTT.
    2. Large-scale Neuroevolution.
    3. Implementing and quantizing a Mamba-style SSM in Spyx.
