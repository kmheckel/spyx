# Surrogate gradients and Gaussian smoothing: two views of one mollification

**Thesis.** The surrogate-gradient trick that makes spiking networks trainable
and the Gaussian-smoothing trick that makes evolution strategies (ES) work are
the *same operation* — convolving a Heaviside step with a bell-shaped kernel and
differentiating the result — applied at two different levels of the computational
graph. Surrogate gradients smooth **each neuron's spike in activation space**; ES
smooths **the whole loss in parameter space**. Seeing them as one thing explains
why the surrogate's *shape* barely matters, tells you exactly what its free
parameter means, and grounds the hybrid trainer in
[`spyx.experimental.hybrid`](../reference/experimental.md).

This page is conceptual (Diátaxis *explanation*). For the taxonomy of training
methods see [Training methods](training-methods.md); for the hybrid that exploits
this connection see the [hybrid module](../reference/experimental.md).

---

## Leg 1 — ES computes the gradient of a Gaussian-smoothed loss

Given any loss $L(\theta)$, define its **Gaussian-smoothed** version by convolving
with an isotropic Gaussian of width $\sigma$:

$$
L_\sigma(\theta) \;=\; \mathbb{E}_{\varepsilon\sim\mathcal N(0,I)}\big[L(\theta+\sigma\varepsilon)\big]
\;=\; (L * \varphi_\sigma)(\theta).
$$

$L_\sigma$ is smooth **even when $L$ is discontinuous**, and $L_\sigma \to L$ as
$\sigma \to 0$ (a mollifier). Its gradient obeys the score-function / Stein
identity

$$
\nabla L_\sigma(\theta)
= \tfrac1\sigma\,\mathbb{E}\big[L(\theta+\sigma\varepsilon)\,\varepsilon\big]
= \mathbb{E}\!\left[\frac{L(\theta+\sigma\varepsilon)-L(\theta-\sigma\varepsilon)}{2\sigma}\,\varepsilon\right],
$$

and the right-hand Monte-Carlo estimator is precisely **antithetic evolution
strategies** (the estimator in `spyx.experimental.hybrid.es_gradient`). So ES is
not a heuristic: it is an unbiased estimate of the gradient of the
Gaussian-smoothed loss. This is the Nesterov–Spokoiny / Salimans view of ES.

## Leg 2 — smoothing the Heaviside spike *is* a sigmoid, and its derivative *is* the surrogate

Now do the same convolution one level down, at a single neuron. Let $v = u-\vartheta$
be membrane potential minus threshold, so the spike is $s = H(v)$ with $H$ the
Heaviside step. Convolve $H$ with a noise kernel $k$ of scale $\sigma$:

$$
H_\sigma(v) \;=\; \mathbb{E}_{\eta\sim k_\sigma}\big[H(v+\eta)\big]
\;=\; \Pr\!\big[\eta > -v\big] \;=\; F_{k_\sigma}(v),
$$

the **CDF of the noise** — an S-curve, i.e. a *sigmoid*. Its derivative is the
noise **density**:

$$
H_\sigma'(v) \;=\; k_\sigma(v),
$$

a bump centred at the threshold. This is exactly the surrogate-gradient recipe:
the forward pass keeps the hard step $H(v)$, the backward pass multiplies by a
bump $k_\sigma(v)$. **Every named surrogate is the density of a noise
distribution, and the "smooth spike" it implies is that noise's CDF.** In the
stochastic-neuron reading this kernel is the *escape-noise* firing-probability
function: $\Pr[\text{spike}] = F_{k_\sigma}(v)$.

### The `spyx.axn` surrogates, read as smoothing kernels

Each factory in [`spyx.axn`](../reference/axn.md) exposes a backward bump
$g(x)$. Identifying $g$ with a noise density $k_\sigma$ names the implicit
smoothing kernel; the sharpness $k$ is an **inverse smoothing width**,
$k \sim 1/\sigma$.

| `spyx.axn` factory | surrogate derivative $g(x)$ | implied noise kernel $k_\sigma$ | smoothed spike (CDF) |
| --- | --- | --- | --- |
| `tanh(k)` | $\operatorname{sech}^2(kx)$ | logistic / hyperbolic-secant | logistic sigmoid $\tfrac12(1+\tanh kx)$ |
| `arctan(k)` | $\dfrac{1}{\pi\,(1+(k\pi x)^2)}$ | Cauchy / Lorentzian (heavy-tailed) | $\tfrac1\pi\arctan(k\pi x)+\tfrac12$ |
| `superspike(k)` | $\dfrac{1}{(1+k\lvert x\rvert)^2}$ | fast-sigmoid (Zenke–Ganguli) | fast sigmoid $\tfrac12\!\left(1+\tfrac{kx}{1+k\lvert x\rvert}\right)$ |
| `triangular(k)` | $\max(0,\,1-\lvert kx\rvert)$ | triangular (sum of two uniforms) | piecewise-quadratic S-curve |
| `boxcar(w,h)` | $h\,\mathbb{1}[\lvert x\rvert\le w/2]$ | uniform on $[-w/2,\,w/2]$ | hard-sigmoid (clipped ramp) |
| `heaviside` / STE | $1$ (identity backward) | improper uniform on $\mathbb R$ | degenerate: infinite-width smoothing |
| *(Gaussian)* | $\propto e^{-x^2/2\sigma^2}$ | Gaussian | probit $\Phi(x/\sigma)$ |

The straight-through estimator is the $\sigma\to\infty$ corner: smooth so hard
that the density is flat and the backward pass is the identity. The Gaussian row
is the one that makes Leg 1 and Leg 2 *literally the same kernel*.

---

## The bridge — and the asymmetry that matters

Both methods defeat the same obstacle (the derivative of $H$ is zero almost
everywhere, or a Dirac at the threshold) by the same means (convolve with a
kernel, differentiate the smooth result). They differ in **where** the noise is
injected and **how** the gradient is taken:

|  | Surrogate gradient | Evolution strategies |
| --- | --- | --- |
| noise injected into | each neuron's pre-activation $v$ (**activation space**) | parameters $\theta$ (**parameter space**) |
| kernel | `spyx.axn` bump ($\operatorname{sech}^2$, Cauchy, box, …) | Gaussian $\varphi_\sigma$ |
| gradient computed by | analytic backprop through the smoothed step | Monte-Carlo sampling (score-function) |
| forward/backward | **inconsistent** — hard forward, soft backward | **consistent** — perturbs the actual forward |
| cost / variance | one pass, deterministic, cheap | $2K$ passes, unbiased, high-variance |

There is one asymmetry worth stating sharply, because it is the crux. **ES is the
gradient of an honest, global smoothed loss** $L_\sigma(\theta)$. The
multi-layer surrogate gradient, by contrast, is *not* the gradient of any global
loss: it coincides with a derivative of the escape-noise firing probability only
**per neuron**, and composing these per-neuron smoothings through the network does
not integrate back to a single scalar objective (Gygax & Zenke, 2025). The
surrogate is a *locally-mollified pseudo-gradient*; ES is a *globally-mollified
true gradient*.

They agree only in a degenerate case: if the map $\theta \mapsto v$ were affine
and the kernels were matched, parameter-space smoothing would push forward exactly
to activation-space smoothing and the two gradients would coincide. Real networks
are deep and nonlinear, so they diverge — **and that divergence is precisely the
surrogate's bias.**

### Why this grounds the Spyx hybrid trainer

The [hybrid method](training-methods.md) reads directly off the
asymmetry. With both quantities estimating (aspects of) the *same* smoothed
landscape:

- $g_s$ — the surrogate gradient — is the cheap, deterministic, *locally*-smoothed
  pseudo-gradient (biased: it is not a true $\nabla L_\sigma$).
- $g_{es}$ — the ES estimate — is the noisy but *globally*-smoothed true gradient
  $\nabla L_\sigma$.
- $g_{\text{orth}} = g_{es} - \langle g_{es}, \hat g_s\rangle \hat g_s$ — the ES
  estimate with its surrogate-aligned part removed — is an estimate of exactly the
  component of the true smoothed gradient that the local surrogate *cannot see*:
  its bias.

So "surrogate bulk direction + orthogonal ES correction" is not an arbitrary
blend — it is "cheap local mollification, corrected in the subspace where local
and global mollification disagree." (The empirical caveat from the study stands:
the correction is high-variance and only pays off when the surrogate's bias is
large; see the [hybrid study](https://github.com/kmheckel/spyx/tree/main/research/new/hybrid_evo_surrogate).)

---

## What the unified view predicts

- **Shape-robustness.** Any normalized bump is a valid mollifier, and to leading
  order all mollifiers of the same width give the same smoothed landscape. So the
  *shape* of the surrogate should be nearly irrelevant and only its *width* should
  matter — which is exactly the striking empirical finding of Zenke & Vogels
  (2021). The smoothing view predicts it rather than merely observing it.
- **The free parameter is a resolution knob.** The sharpness $k$ in `spyx.axn`
  is an inverse smoothing scale $k\sim 1/\sigma$: too sharp ($k\to\infty$) removes
  the smoothing and reinstates the dead Heaviside derivative (vanishing-gradient
  regions); too broad over-smooths and biases. Same bias/resolution trade-off as
  choosing $\sigma$ in ES.
- **Annealing = graduated non-convexity.** Scheduling the surrogate from broad to
  sharp is continuation/graduated-non-convexity on the mollified loss — a
  principled reason to anneal $k$ (equivalently $\sigma$) during training.
- **The exact-coincidence object already ships.** In a *stochastic* SNN where the
  spike is literally $\mathrm{Bernoulli}(F(v))$, the expected output is the sigmoid
  $F(v)$ and the surrogate gradient becomes the **exact** gradient of the expected
  loss — no approximation. That is
  [`spyx.experimental.stochastic.sigmoid_bernoulli`](../reference/experimental.md)
  / `SPSN`: the object where Leg 1 and Leg 2 collapse into one and the surrogate
  stops being a surrogate.

---

## Honest placement in the literature

!!! note "What is established vs. what this page adds"
    **Both legs are known results.** ES as the gradient of a Gaussian-smoothed
    loss is standard (Nesterov & Spokoiny 2017; Salimans et al. 2017). The
    surrogate-derivative-as-smoothed/stochastic-firing-function is established, and
    the subtle point that multi-layer surrogate gradients are *not* the gradient of
    any surrogate loss is due to Gygax & Zenke (2025). Framing surrogate gradients
    as adaptive smoothing also appears in Wang et al. (2023).

    **What this note draws together** — and what we have not seen stated explicitly
    in the SNN literature — is the *bridge*: that the surrogate-derivative bump and
    the ES perturbation density are the **same mollification kernel** acting at two
    different levels (per-neuron activation space vs. global parameter space), that
    this makes surrogate-GD and ES the deterministic-local and stochastic-global
    gradients of one smoothing operation, and that the Gygax–Zenke asymmetry (ES
    smooths a real loss; the surrogate does not) is exactly what a hybrid
    orthogonal-ES correction should target. We state this as a conceptual synthesis
    and an invitation to be corrected, **not** a novelty claim on either leg.

### References

- Y. Nesterov, V. Spokoiny (2017). *Random Gradient-Free Minimization of Convex
  Functions.* Foundations of Computational Mathematics.
- T. Salimans, J. Ho, X. Chen, S. Sidor, I. Sutskever (2017). *Evolution
  Strategies as a Scalable Alternative to Reinforcement Learning.*
  arXiv:1703.03864.
- E. O. Neftci, H. Mostafa, F. Zenke (2019). *Surrogate Gradient Learning in
  Spiking Neural Networks.* IEEE Signal Processing Magazine. arXiv:1901.09948.
- F. Zenke, S. Ganguli (2018). *SuperSpike: Supervised Learning in Multilayer
  Spiking Neural Networks.* Neural Computation.
- F. Zenke, T. P. Vogels (2021). *The Remarkable Robustness of Surrogate Gradient
  Learning for Instilling Complex Function in Spiking Neural Networks.* Neural
  Computation.
- J. Gygax, F. Zenke (2025). *Elucidating the Theoretical Underpinnings of
  Surrogate Gradient Learning in Spiking Neural Networks.* Neural Computation.
  arXiv:2404.14964.
- Y. Wang et al. (2023). *Adaptive Smoothing Gradient Learning for Spiking Neural
  Networks.* ICML.
- Y. Bengio, N. Léonard, A. Courville (2013). *Estimating or Propagating Gradients
  Through Stochastic Neurons.* arXiv:1308.3432 (straight-through estimator).
- N. Maheswaranathan, L. Metz, G. Tucker, D. Choi, J. Sohl-Dickstein (2019).
  *Guided Evolutionary Strategies.* arXiv:1806.10230 (the in-subspace complement of
  the orthogonal correction).

### See also in Spyx

- [`spyx.axn`](../reference/axn.md) — the surrogate kernels tabulated above.
- [`spyx.experimental.hybrid`](../reference/experimental.md) — surrogate + orthogonal-ES.
- [`spyx.experimental.stochastic`](../reference/experimental.md) — `SPSN`, the exact-coincidence object.
- [Training methods](training-methods.md) · [Choosing an approach](choosing-an-approach.md).
