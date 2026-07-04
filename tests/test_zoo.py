"""Tests for the spyx.experimental.zoo recipe registry.

All recipes run on synthetic data (no downloads) and are sized for a few
seconds total on CPU. Not marked ``network``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from spyx.experimental.zoo import REGISTRY, Recipe, get, list_recipes

# Gradient-trained recipes whose demo loss should decrease.
_GRADIENT_RECIPES = {"classification-rsnn", "language-s5"}


def _finite(values) -> bool:
    return bool(jnp.all(jnp.isfinite(jnp.asarray(values))))


def test_registry_spans_three_applications():
    assert len(REGISTRY) >= 3
    applications = {r.application for r in REGISTRY.values()}
    assert applications == {"control", "classification", "language"}


def test_registry_fields_populated():
    for name, recipe in REGISTRY.items():
        assert isinstance(recipe, Recipe)
        assert recipe.name == name
        for field in ("name", "application", "method", "architecture", "describe"):
            value = getattr(recipe, field)
            assert isinstance(value, str) and value, f"{name}.{field} empty"
        assert callable(recipe.build)
        assert callable(recipe.synthetic_batch)
        assert callable(recipe.loss)
        assert callable(recipe.demo)


def test_helpers():
    assert get("control-lif-es").application == "control"
    with pytest.raises(KeyError):
        get("does-not-exist")
    assert len(list_recipes()) == len(REGISTRY)
    assert {r.name for r in list_recipes(application="language")} == {"language-s5"}
    assert {r.name for r in list_recipes(method="surrogate")} == {"classification-rsnn"}
    assert {r.name for r in list_recipes(method="gradient")} == {"language-s5"}
    assert list_recipes(application="control", method="surrogate") == []


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_build_and_loss_finite(name):
    recipe = get(name)
    model = recipe.build(nnx.Rngs(0))
    assert isinstance(model, nnx.Module)
    batch = recipe.synthetic_batch(jax.random.PRNGKey(1))
    assert isinstance(batch, tuple)
    value = recipe.loss(model, *batch)
    assert _finite(value)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_demo_runs_and_history_finite(name):
    recipe = get(name)
    history = recipe.demo(steps=6)
    assert len(history) == 6
    assert _finite(history)


@pytest.mark.parametrize("name", sorted(_GRADIENT_RECIPES))
def test_gradient_recipes_improve(name):
    recipe = get(name)
    history = recipe.demo(steps=25)
    # Loss at the end should be no worse than at the start (small tolerance).
    assert history[-1] <= history[0] + 1e-4
