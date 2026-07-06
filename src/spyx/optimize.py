"""High-level training utilities for Spyx SNNs.

Issue #26 asked for a "quick train/eval loop" so users don't have to
re-derive the ``nnx.Optimizer`` + ``nnx.value_and_grad`` + per-epoch boiler-
plate every time they build a new model. This module provides that, with a
minimum of magic:

* :func:`train_step` — JIT-compiled single-step update.
* :func:`eval_step` — JIT-compiled single-step accuracy/loss.
* :func:`fit` — end-to-end Python epoch loop that iterates an iterable data
  source (anything yielding ``(events, targets)`` tuples — Spyx loader,
  generator, or plain list).

The utilities deliberately don't hide the loss / metric / optimizer choices.
Pass your own via ``spyx.fn.integral_crossentropy`` / ``optax.lion`` etc.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

History = list[dict[str, float]]


def make_train_step(
    loss_fn: Callable[[Any], jax.Array],
) -> Callable[..., jax.Array]:
    """Build a JIT-compiled single-step updater.

    The returned callable has signature ``(model, optimizer, *loss_args) ->
    loss_value`` and mutates ``model`` / ``optimizer`` in place via NNX.

    :param loss_fn: closure taking ``(model, *loss_args)`` and returning a
        scalar loss. Typically wraps ``spyx.fn.integral_crossentropy()``.
    """

    @nnx.jit
    def step(model, optimizer, *loss_args):
        def _loss(m):
            return loss_fn(m, *loss_args)

        loss, grads = nnx.value_and_grad(_loss)(model)
        optimizer.update(model, grads)
        return loss

    return step


def make_eval_step(
    metric_fn: Callable[[Any], tuple[jax.Array, jax.Array]],
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build a JIT-compiled single-step evaluation callable.

    :param metric_fn: closure taking ``(model, *metric_args)`` and returning
        ``(accuracy_or_similar, loss)``.
    """

    @nnx.jit
    def step(model, *metric_args):
        return metric_fn(model, *metric_args)

    return step


def fit(
    model: nnx.Module,
    tx: optax.GradientTransformation,
    loss_fn: Callable[[Any], jax.Array],
    train_iter: Callable[[], Iterable[tuple[Any, ...]]],
    *,
    epochs: int,
    eval_iter: Callable[[], Iterable[tuple[Any, ...]]] | None = None,
    eval_fn: Callable[..., tuple[jax.Array, jax.Array]] | None = None,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> History:
    """End-to-end training loop.

    :param model: the Spyx / Flax NNX module to train.
    :param tx: an Optax :class:`GradientTransformation` (e.g. ``optax.lion(3e-4)``).
    :param loss_fn: ``(model, *batch) -> loss``. ``batch`` is whatever
        ``train_iter`` yields.
    :param train_iter: zero-arg callable returning a fresh iterable of
        training batches each epoch. This matches the ``spyx.data.*_loader``
        convention where ``loader.train_epoch()`` is called per epoch.
    :param epochs: number of training epochs.
    :param eval_iter: optional zero-arg callable yielding evaluation batches.
    :param eval_fn: optional ``(model, *batch) -> (accuracy, loss)``;
        required if ``eval_iter`` is set.
    :param on_epoch_end: optional callback ``(epoch, metrics_dict) -> None``
        for progress printing etc. Metrics dict carries keys
        ``train_loss``, plus ``eval_acc`` / ``eval_loss`` when evaluating.
    :return: list of per-epoch metric dicts.
    """
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    train_step = make_train_step(loss_fn)
    eval_step = make_eval_step(eval_fn) if eval_fn is not None else None

    def _mean_or_raise(xs, *, kind: str, epoch: int):
        # jnp.stack(()) raises a low-signal error; produce a clearer one. This
        # most commonly fires when a Grain loader's `drop_remainder=True`
        # combined with `batch_size > dataset_size` yields a zero-batch epoch.
        if not xs:
            raise RuntimeError(
                f"fit(): {kind}_iter() yielded no batches at epoch {epoch}. "
                "Check that batch_size <= dataset size and drop_remainder=False "
                "if you want partial trailing batches."
            )
        return float(jnp.mean(jnp.stack(xs)))

    history: History = []
    for epoch in range(epochs):
        train_losses = []
        for batch in train_iter():
            batch_args = batch if isinstance(batch, tuple) else (batch,)
            train_losses.append(train_step(model, optimizer, *batch_args))
        metrics: dict[str, float] = {
            "train_loss": _mean_or_raise(train_losses, kind="train", epoch=epoch)
        }

        if eval_iter is not None and eval_step is not None:
            accs, losses = [], []
            for batch in eval_iter():
                batch_args = batch if isinstance(batch, tuple) else (batch,)
                acc, loss = eval_step(model, *batch_args)
                accs.append(acc)
                losses.append(loss)
            metrics["eval_acc"] = _mean_or_raise(accs, kind="eval", epoch=epoch)
            metrics["eval_loss"] = _mean_or_raise(losses, kind="eval", epoch=epoch)

        history.append(metrics)
        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)

    return history


class SolverImpl(NamedTuple):
    """The three pure functions a solver hands to :func:`compile_fit`.

    - ``init(key) -> state`` — build the optimiser state from the initial params.
    - ``step(state, batch, key) -> (state, metric)`` — one update; ``metric`` is a
      scalar summarised into ``history`` (the loss / best-fitness).
    - ``get_params(state) -> params`` — the current point estimate (the ``mean``
      for ask/tell strategies).
    """

    init: Callable[[jax.Array], Any]
    step: Callable[[Any, tuple, jax.Array], tuple[Any, jax.Array]]
    get_params: Callable[[Any], Any]


# A Solver is a *builder* ``(loss_flat, params0) -> SolverImpl``, where
# ``loss_flat(params, batch) -> scalar`` is supplied by :func:`compile_fit` (it
# closes over the model's graph). Building at bind time lets ES/CMA solvers set up
# ``ravel_pytree`` / strategy state against the concrete parameter structure.
Solver = Callable[[Callable[[Any, tuple], jax.Array], Any], SolverImpl]


def backprop(tx: optax.GradientTransformation) -> Solver:
    """A plain gradient-descent :data:`Solver` — ``value_and_grad`` + an Optax update.

    This is the surrogate-gradient path; passing an Optax ``GradientTransformation``
    straight to :func:`compile_fit` wraps it in this automatically.
    """

    def build(loss_flat, params0):
        def init(key):
            return (params0, tx.init(params0))

        def step(state, batch, key):
            params, opt_state = state
            loss, grads = jax.value_and_grad(lambda p: loss_flat(p, batch))(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        return SolverImpl(init, step, lambda s: s[0])

    return build


def compile_fit(
    model: nnx.Module,
    solver: "Solver | optax.GradientTransformation",
    loss_fn: Callable[..., jax.Array],
    train_data: tuple[Any, ...],
    *,
    epochs: int,
    eval_data: tuple[Any, ...] | None = None,
    metric_fn: Callable[..., jax.Array] | None = None,
    key: jax.Array | None = None,
) -> tuple[nnx.Module, dict[str, jax.Array]]:
    r"""Compile the *entire* training loop to a single XLA dispatch.

    Where :func:`fit` is a Python epoch loop driving a per-step JIT, this stages the
    whole dataset on-device and ``jax.lax.scan``\ s the loop over epochs × batches
    under one ``jax.jit`` — a full run becomes a single compiled kernel with no
    per-step Python or re-tracing (the throughput pattern the Spyx paper relies on).

    The optimiser is a :data:`Solver`, so gradient descent and gradient-free
    ask/tell (CMA-ES etc.) share the same compiled loop: pass an Optax
    ``GradientTransformation`` (wrapped in :func:`backprop`), or a solver from
    :mod:`spyx.experimental.evolve`.

    :param model: the SNN to train (any Flax NNX module).
    :param solver: a :data:`Solver` builder, or an Optax ``GradientTransformation``
        (e.g. ``optax.adam(3e-3)``) which is treated as ``backprop(tx)``.
    :param loss_fn: ``(model, *batch) -> scalar``.
    :param train_data: ``(X, Y, …)`` staged with a leading batch axis to scan over —
        each leaf shaped ``[n_batches, batch, ...]``, already on device (``jnp.stack``
        your loader's batches once).
    :param epochs: number of passes over the staged batches.
    :param eval_data: optional held-out data staged the same way; scored with
        ``metric_fn`` once per epoch *inside* the compiled loop.
    :param metric_fn: ``(model, *batch) -> scalar`` (e.g. accuracy); required with
        ``eval_data``.
    :param key: PRNG key threaded through ``init`` / ``step`` (ES, dropout, …).
    :return: ``(trained_model, history)`` where ``history`` holds stacked per-epoch
        arrays: ``train_loss`` (the mean per-step metric) and ``eval_metric``.
    """
    if eval_data is not None and metric_fn is None:
        raise ValueError("metric_fn is required when eval_data is given")

    graphdef, params0, rest = nnx.split(model, nnx.Param, ...)

    def loss_flat(params, batch):
        return loss_fn(nnx.merge(graphdef, params, rest), *batch)

    if isinstance(solver, optax.GradientTransformation):
        solver = backprop(solver)
    impl = solver(loss_flat, params0)

    @jax.jit
    def _run(key):
        k_init, k_run = jax.random.split(key)
        state0 = impl.init(k_init)

        def step(carry, batch):
            state, key = carry
            key, sub = jax.random.split(key)
            state, metric = impl.step(state, batch, sub)
            return (state, key), metric

        def epoch(carry, _):
            carry, metrics = jax.lax.scan(step, carry, train_data)
            rec = {"train_loss": jnp.mean(metrics)}
            if eval_data is not None:
                assert metric_fn is not None  # guaranteed by the guard above
                mfn = metric_fn
                m = nnx.merge(graphdef, impl.get_params(carry[0]), rest)
                rec["eval_metric"] = jnp.mean(
                    jax.vmap(lambda *b: mfn(m, *b))(*eval_data)
                )
            return carry, rec

        (final, _k), history = jax.lax.scan(epoch, (state0, k_run), None, length=epochs)
        return impl.get_params(final), history

    final_params, history = _run(jax.random.PRNGKey(0) if key is None else key)
    return nnx.merge(graphdef, final_params, rest), history


__all__ = [
    "fit",
    "compile_fit",
    "backprop",
    "SolverImpl",
    "Solver",
    "make_train_step",
    "make_eval_step",
]
