"""The Spyx recipe **zoo** — runnable, synthetic-data SNN recipes.

**Experimental.** This whole subpackage lives under
:mod:`spyx.experimental`; its API may change without a deprecation cycle.

Each recipe is a self-contained, download-free example of training a spiking /
state-space model for one *application*, tagged by *method* × *architecture*:

============== ============== ============= ==============
application    method         architecture module
============== ============== ============= ==============
control        evolutionary   LIF-MLP       :mod:`.control`
classification surrogate      RSNN          :mod:`.classification`
language       surrogate      S5            :mod:`.language`
============== ============== ============= ==============

Every recipe exposes the same small surface via a :class:`Recipe` record:

* ``build(rngs) -> nnx.Module`` — construct the model.
* ``synthetic_batch(...) -> tuple`` — sample a download-free batch.
* ``loss(model, *batch) -> scalar`` — a finite objective on that batch.
* ``demo(steps=...) -> list[float]`` — run a few train/evolve steps and return
  the fitness/loss history.

The zoo is importable as its own subpackage — ``from spyx.experimental.zoo
import REGISTRY, list_recipes, get`` — without touching
``spyx.experimental.__init__``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from flax import nnx

from . import classification, control, language


@dataclass(frozen=True)
class Recipe:
    """A single runnable recipe, keyed by application and tagged by method × arch.

    :name: unique registry key.
    :application: one of ``'control'``, ``'classification'``, ``'language'``.
    :method: training method, e.g. ``'evolutionary'``, ``'surrogate'``,
        ``'conversion'``, ``'hybrid'``.
    :architecture: model family, e.g. ``'LIF-MLP'``, ``'RSNN'``, ``'S5'``.
    :build: ``(nnx.Rngs) -> nnx.Module`` model constructor.
    :synthetic_batch: ``(...) -> tuple`` download-free batch sampler; the tuple
        is splatted into ``loss`` after the model.
    :describe: one-line human-readable description.
    :loss: ``(model, *batch) -> scalar`` finite objective (fitness for
        evolutionary recipes, training loss for gradient recipes).
    :demo: ``(steps=...) -> list[float]`` short run returning a fitness/loss
        history.
    """

    name: str
    application: str
    method: str
    architecture: str
    build: Callable[[nnx.Rngs], nnx.Module]
    synthetic_batch: Callable[..., tuple]
    describe: str
    loss: Callable[..., object]
    demo: Callable[..., list[float]]


def _recipe_from_module(module) -> Recipe:
    """Assemble a :class:`Recipe` from a recipe module's module-level surface."""
    return Recipe(
        name=module.NAME,
        application=module.APPLICATION,
        method=module.METHOD,
        architecture=module.ARCHITECTURE,
        build=module.build,
        synthetic_batch=module.synthetic_batch,
        describe=module.DESCRIBE,
        loss=module.loss,
        demo=module.demo,
    )


REGISTRY: dict[str, Recipe] = {
    recipe.name: recipe
    for recipe in (
        _recipe_from_module(control),
        _recipe_from_module(classification),
        _recipe_from_module(language),
    )
}


def list_recipes(
    application: str | None = None, method: str | None = None
) -> list[Recipe]:
    """Return recipes, optionally filtered by ``application`` and/or ``method``.

    :application: keep only recipes with this application (``None`` = any).
    :method: keep only recipes with this method (``None`` = any).
    :return: list of matching :class:`Recipe` records.
    """
    return [
        recipe
        for recipe in REGISTRY.values()
        if (application is None or recipe.application == application)
        and (method is None or recipe.method == method)
    ]


def get(name: str) -> Recipe:
    """Look up a recipe by name, raising ``KeyError`` with the valid keys."""
    if name not in REGISTRY:
        raise KeyError(f"unknown recipe {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = [
    "Recipe",
    "REGISTRY",
    "list_recipes",
    "get",
    "control",
    "classification",
    "language",
]
