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

from typing import Any, Callable, Iterable

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


__all__ = ["fit", "make_train_step", "make_eval_step"]
