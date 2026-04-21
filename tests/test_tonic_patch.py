"""Regression tests for the SHD loader + HSD monkey-patch shipped in spyx.data.

Two protections are covered:

1. The monkey-patch on ``tonic.datasets.hsd.HSD.__getitem__`` drops
   non-finite timestamps instead of letting them cast to INT_MIN / 0.

2. The SHD transform pipeline (``Downsample`` → ``ToFrame``) actually
   produces non-empty frames. Previously ``Downsample(time_factor=1e-6 /
   (1/sample_T))`` pre-compressed timestamps into [0, sample_T], which
   then collapsed under tonic's integer floor-division slicing and
   silently zeroed every frame.

Skipped cleanly when tonic isn't installed.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("tonic")
pytest.importorskip("h5py")


def _write_fake_shd(path: str) -> None:
    """Build a minimal SHD-shaped HDF5 with two samples, one carrying NaN/inf."""
    import h5py

    with h5py.File(path, "w") as f:
        # Ragged arrays: SHD stores spikes/times and spikes/units as
        # length-N lists where each element is a 1-D array of spike
        # times for one sample. h5py uses vlen for this.
        vlen_float = h5py.vlen_dtype(np.dtype("float32"))
        vlen_int = h5py.vlen_dtype(np.dtype("int64"))

        times = f.create_dataset("spikes/times", (2,), dtype=vlen_float)
        units = f.create_dataset("spikes/units", (2,), dtype=vlen_int)

        # Sample 0: well-formed.
        times[0] = np.array([0.01, 0.05, 0.2], dtype=np.float32)
        units[0] = np.array([10, 42, 100], dtype=np.int64)

        # Sample 1: contains NaN and +inf, which the unpatched loader
        # propagates into the int-typed `t` field.
        times[1] = np.array([0.1, np.nan, np.inf, 0.3], dtype=np.float32)
        units[1] = np.array([5, 6, 7, 8], dtype=np.int64)

        # Labels: ignored by the monkey-patch but required by the schema.
        f.create_dataset("labels", data=np.array([0, 1], dtype=np.int64))
        f.create_dataset("extra/keys", data=np.array([b"zero", b"one"]))
        f.create_dataset("extra/speaker", data=np.array([0, 0]))


def test_hsd_monkey_patch_drops_non_finite_entries():
    # Importing spyx.data is what applies the monkey-patch. The `noqa: F401`
    # keeps ruff from stripping this import as unused.
    from tonic.datasets.hsd import HSD

    import spyx.data  # noqa: F401

    # Monkey-patch marker lets us confirm spyx.data applied its version.
    assert getattr(HSD.__getitem__, "_spyx_patched", False), (
        "spyx.data should have monkey-patched HSD.__getitem__ at import time; "
        "_spyx_patched sentinel is missing"
    )

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = os.path.join(tmp, "shd_train.h5")
        _write_fake_shd(h5_path)

        # Build a fresh HSD instance pointing at the synthetic file.
        # HSD.__init__ expects the SHD subclass machinery; easiest to
        # instantiate HSD's bare contract manually.
        class _HSDStub(HSD):
            def __init__(self_):
                self_.location_on_system = tmp
                self_.data_filename = "shd_train.h5"
                self_.transform = None
                self_.target_transform = None

        ds = _HSDStub()

        # Sample 0: clean. All three spikes survive.
        events0, _ = ds[0]
        assert events0.shape[0] == 3
        assert np.all(np.isfinite(events0["t"]))

        # Sample 1: NaN + inf are dropped; only 2 spikes (0.1s, 0.3s) remain.
        events1, _ = ds[1]
        assert events1.shape[0] == 2, (
            f"expected 2 finite spikes after dropping NaN/inf, got {events1.shape[0]}"
        )
        assert np.all(np.isfinite(events1["t"]))
        # The finite spikes should come back as 100000 and 300000 microseconds.
        assert sorted(events1["t"].tolist()) == [100000, 300000]


def test_shd_transform_pipeline_produces_nonempty_frames():
    """The Downsample → ToFrame pipeline must actually bin events into frames.

    Regression for the interaction bug between ``Downsample(time_factor=...)``
    and ``ToFrame(n_time_bins=...)``: pre-scaling timestamps into [0, T]
    causes tonic's SliceByTimeBins to floor-divide the per-bin window to
    zero and produce all-zero frames. spyx.data.SHD_loader now leaves
    timestamps in microseconds so ToFrame gets a wide enough range.
    """
    from tonic import transforms

    # Synthetic SHD-shaped sample: 1000 events spanning ~1 second.
    rng = np.random.default_rng(0)
    events = np.zeros(
        1000, dtype=[("t", int), ("x", int), ("p", int)]
    )
    events["t"] = np.sort(rng.uniform(0, 1e6, 1000).astype(int))
    events["x"] = rng.integers(0, 700, 1000)
    events["p"] = 1

    net_channels = 128
    sample_T = 128

    # Reproduce the fixed pipeline from spyx.data.SHD_loader.
    pipeline = transforms.Compose([
        transforms.Downsample(spatial_factor=net_channels / 700),
        transforms.ToFrame(
            sensor_size=(net_channels, 1, 1),
            n_time_bins=sample_T,
        ),
    ])
    frame = pipeline(events)
    assert frame.sum() > 0, (
        "SHD transform pipeline collapsed events to empty frames; "
        "check that time_factor is NOT being applied before ToFrame."
    )
    # A typical 1000-event sample should spread over most time bins.
    time_axis_totals = frame.sum(axis=(1, 2))
    assert (time_axis_totals > 0).sum() > sample_T // 4, (
        f"expected spikes in > {sample_T // 4} time bins, "
        f"got {(time_axis_totals > 0).sum()}"
    )
