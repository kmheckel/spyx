"""Tests for spyx.raven (Routing Slot Memory)."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from spyx import raven

# ---------------------------------------------------------------------------
# shapes / dtypes / basic run
# ---------------------------------------------------------------------------


def test_forward_shape_and_dtype():
    rngs = nnx.Rngs(0)
    block = raven.RavenRSM(d_model=8, n_slots=6, d_slot=5, rngs=rngs)
    u = jax.random.normal(jax.random.PRNGKey(0), (10, 3, 8))
    y = block(u)
    assert y.shape == (10, 3, 8)
    assert y.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(y))


def test_initial_state_zeros():
    rngs = nnx.Rngs(0)
    block = raven.RavenRSM(d_model=4, n_slots=3, d_slot=7, rngs=rngs)
    s = block.initial_state(5)
    assert s.shape == (5, 3, 7)
    assert jnp.all(s == 0.0)


def test_default_d_slot_is_d_model():
    rngs = nnx.Rngs(0)
    block = raven.RavenRSM(d_model=8, n_slots=4, rngs=rngs)
    assert block.d_slot == 8
    assert block.initial_state(2).shape == (2, 4, 8)


def test_decay_in_unit_interval():
    rngs = nnx.Rngs(0)
    block = raven.RavenRSM(d_model=8, n_slots=4, d_slot=4, rngs=rngs)
    a = block.decay
    assert a.shape == (4, 4)
    assert jnp.all(a > 0.0) and jnp.all(a < 1.0)


def test_step_matches_scan():
    """The single-step API must reproduce the batched scan path exactly."""
    rngs = nnx.Rngs(1)
    block = raven.RavenRSM(d_model=6, n_slots=4, d_slot=3, rngs=rngs)
    T, B = 7, 2
    u = jax.random.normal(jax.random.PRNGKey(3), (T, B, 6))

    y_batched = block(u)

    s = block.initial_state(B)
    ys = []
    for t in range(T):
        s, y_t = block.step(s, u[t])
        ys.append(y_t)
    y_seq = jnp.stack(ys, axis=0)

    assert jnp.allclose(y_batched, y_seq, atol=1e-5)


# ---------------------------------------------------------------------------
# dense-router reduction: r_t == all-ones  ->  gated diagonal recurrence
# ---------------------------------------------------------------------------


def test_dense_router_reduces_to_gated_diagonal_ssm():
    """With r_t forced to all-ones the RSM is S_t = a ⊙ S_{t-1} + U_t."""
    rngs = nnx.Rngs(2)
    block = raven.RavenRSM(d_model=6, n_slots=5, d_slot=4, rngs=rngs)
    T, B = 9, 3
    u = jax.random.normal(jax.random.PRNGKey(4), (T, B, 6))

    ones = jnp.ones((T, B, block.n_slots))
    y_dense = block._run(u, ones)

    # Independent reference: plain gated diagonal recurrence per slot.
    U = block.write(u).reshape(T, B, block.n_slots, block.d_slot)
    attn = jax.nn.softmax(block.readout_query(u), axis=-1)
    a = block.decay  # (M, d_slot)

    s = block.initial_state(B)
    reads = []
    for t in range(T):
        s = a[None] * s + U[t]  # r == 1 everywhere
        reads.append(jnp.einsum("bm,bmd->bd", attn[t], s))
    read_seq = jnp.stack(reads, axis=0)
    y_ref = block.out_proj(read_seq)

    assert jnp.allclose(y_dense, y_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# sparsity: unselected slots (r_t[m] == 0) are provably unchanged
# ---------------------------------------------------------------------------


def test_topk_router_leaves_unselected_slots_unchanged():
    rngs = nnx.Rngs(5)
    block = raven.RavenRSM(d_model=8, n_slots=6, d_slot=4, hard_top_k=1, rngs=rngs)
    B = 4
    u_t = jax.random.normal(jax.random.PRNGKey(6), (B, 8))

    s_prev = jax.random.normal(jax.random.PRNGKey(7), (B, 6, 4))
    r_t = block._route(u_t)  # (B, M)
    s_new, _ = block.step(s_prev, u_t)

    # top-1 must zero out all but (at most) one slot per row.
    assert jnp.all(jnp.sum(r_t > 0, axis=-1) <= 1)
    # exactly-zero-gate slots must be byte-for-byte unchanged.
    mask = r_t == 0.0  # (B, M)
    assert jnp.any(mask)  # there are unselected slots to check
    m3 = mask[..., None]
    unchanged = jnp.where(m3, s_new - s_prev, 0.0)
    assert jnp.max(jnp.abs(unchanged)) == 0.0


# ---------------------------------------------------------------------------
# gradients flow to router, decay, write, readout
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_components():
    rngs = nnx.Rngs(8)
    block = raven.RavenRSM(d_model=6, n_slots=4, d_slot=4, rngs=rngs)
    u = jax.random.normal(jax.random.PRNGKey(9), (8, 2, 6))

    def loss_fn(m):
        return jnp.mean(m(u) ** 2)

    grads = nnx.grad(loss_fn)(block)

    # Flatten grads with paths and build {joined-path: l2-norm}.
    leaves = jax.tree_util.tree_leaves_with_path(grads)
    norms = {}
    for path, leaf in leaves:
        name = "/".join(str(getattr(p, "key", p)) for p in path)
        norms[name] = float(jnp.linalg.norm(leaf))

    for component in ("router", "raw_decay", "write", "readout_query", "out_proj"):
        matched = [n for n, v in norms.items() if component in n and v > 0.0]
        assert matched, f"no non-zero gradient reached {component}: {list(norms)}"


# ---------------------------------------------------------------------------
# recall task generator + learnability
# ---------------------------------------------------------------------------


def test_make_recall_batch_shapes_and_binding():
    key = jax.random.PRNGKey(0)
    n_pairs, n_keys, n_values, batch = 3, 6, 5, 4
    u, target = raven.make_recall_batch(
        key, batch=batch, n_pairs=n_pairs, n_keys=n_keys, n_values=n_values
    )
    T = 2 * n_pairs + 1
    d_model = n_keys + n_values
    assert u.shape == (T, batch, d_model)
    assert target.shape == (batch,)
    # Each token is a one-hot.
    assert jnp.all(jnp.sum(u, axis=-1) == 1.0)
    # The query token (last) is a key-encoding; its id must reappear as an
    # earlier key token, and the bound value must match the target.
    for b in range(batch):
        qid = int(jnp.argmax(u[T - 1, b]))
        assert qid < n_keys  # query reuses a key encoding
        found = False
        for p in range(n_pairs):
            if int(jnp.argmax(u[2 * p, b])) == qid:
                vid = int(jnp.argmax(u[2 * p + 1, b])) - n_keys
                assert vid == int(target[b])
                found = True
        assert found


class _RecallNet(nnx.Module):
    """RavenRSM + a value-classification head reading the final timestep."""

    def __init__(self, d_model, n_values, *, n_slots, d_slot, rngs):
        self.rsm = raven.RavenRSM(d_model, n_slots=n_slots, d_slot=d_slot, rngs=rngs)
        self.head = nnx.Linear(d_model, n_values, rngs=rngs)

    def __call__(self, u):
        y = self.rsm(u)  # (T, B, d_model)
        return self.head(y[-1])  # (B, n_values)


def test_overfits_tiny_recall_task():
    """RavenRSM should overfit a single small MQAR batch well above chance."""
    n_pairs, n_keys, n_values, batch = 2, 4, 4, 8
    d_model = n_keys + n_values

    u, target = raven.make_recall_batch(
        jax.random.PRNGKey(0),
        batch=batch,
        n_pairs=n_pairs,
        n_keys=n_keys,
        n_values=n_values,
    )

    net = _RecallNet(d_model, n_values, n_slots=8, d_slot=8, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(net, optax.adam(5e-3), wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, u, target):
        def loss_fn(m):
            logits = m(u)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, target
            ).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    for _ in range(400):
        step(net, optimizer, u, target)

    logits = net(u)
    acc = float(jnp.mean(jnp.argmax(logits, axis=-1) == target))
    # Chance is 1 / n_values = 0.25; require clearly-learned behaviour.
    assert acc > 0.75, f"recall accuracy did not climb above chance: {acc:.3f}"


# ===========================================================================
# Spiking Raven: SpikingSlotMemory
# ===========================================================================


def test_spiking_forward_shape_and_dtype():
    rngs = nnx.Rngs(0)
    mem = raven.SpikingSlotMemory(d_model=8, n_slots=6, d_slot=5, rngs=rngs)
    u = jax.random.normal(jax.random.PRNGKey(0), (10, 3, 8))
    s = mem(u)
    assert s.shape == (10, 3, 6, 5)
    assert s.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(s))


def test_spiking_default_d_slot_and_initial_state():
    rngs = nnx.Rngs(0)
    mem = raven.SpikingSlotMemory(d_model=8, n_slots=4, rngs=rngs)
    assert mem.d_slot == 8
    v = mem.initial_state(2)
    assert v.shape == (2, 4, 8)
    assert jnp.all(v == 0.0)


def test_spiking_step_matches_scan():
    """The single-step API must reproduce the batched scan path exactly."""
    rngs = nnx.Rngs(1)
    mem = raven.SpikingSlotMemory(d_model=6, n_slots=4, d_slot=3, rngs=rngs)
    T, B = 7, 2
    u = jax.random.normal(jax.random.PRNGKey(3), (T, B, 6))

    s_batched = mem(u)

    v = mem.initial_state(B)
    outs = []
    for t in range(T):
        v, s_t = mem.step(v, u[t])
        outs.append(s_t)
    s_seq = jnp.stack(outs, axis=0)

    assert jnp.array_equal(s_batched, s_seq)


def test_spiking_reuses_ravenrsm_router():
    """Router REUSE: the spiking variant must use the *same* SlotRouter class
    as RavenRSM -- not a divergent fork."""
    rngs = nnx.Rngs(2)
    mem = raven.SpikingSlotMemory(d_model=8, n_slots=5, d_slot=4, rngs=rngs)
    rsm = raven.RavenRSM(d_model=8, n_slots=5, d_slot=4, rngs=rngs)
    assert isinstance(mem.router, raven.SlotRouter)
    assert type(mem.router) is type(rsm.router)
    # _route delegates to that router and yields per-slot gates in [0, 1].
    u_t = jax.random.normal(jax.random.PRNGKey(1), (3, 8))
    r = mem._route(u_t)
    assert r.shape == (3, 5)
    assert jnp.all(r >= 0.0) and jnp.all(r <= 1.0)
    assert jnp.array_equal(r, mem.router(u_t))


def test_spiking_shielding_unrouted_membranes_unchanged():
    """Unrouted slots (r_t[m] == 0) keep their membrane byte-for-byte."""
    rngs = nnx.Rngs(5)
    mem = raven.SpikingSlotMemory(
        d_model=8, n_slots=6, d_slot=4, hard_top_k=1, rngs=rngs
    )
    B = 4
    u_t = jax.random.normal(jax.random.PRNGKey(6), (B, 8))
    v_prev = jax.random.normal(jax.random.PRNGKey(7), (B, 6, 4))

    r_t = mem._route(u_t)  # (B, M)
    v_new, _ = mem.step(v_prev, u_t)

    # top-1 must zero out all but (at most) one slot per row.
    assert jnp.all(jnp.sum(r_t > 0, axis=-1) <= 1)
    mask = r_t == 0.0  # (B, M)
    assert jnp.any(mask)  # there are shielded slots to check
    m3 = mask[..., None]
    unchanged = jnp.where(m3, v_new - v_prev, 0.0)
    assert jnp.max(jnp.abs(unchanged)) == 0.0


def test_spiking_spikes_are_binary_and_gradients_flow():
    rngs = nnx.Rngs(8)
    mem = raven.SpikingSlotMemory(d_model=6, n_slots=4, d_slot=4, rngs=rngs)
    u = jax.random.normal(jax.random.PRNGKey(9), (8, 2, 6))

    s = mem(u)
    # Forward spikes are Heaviside -> in the surrogate's {0, 1} range.
    assert jnp.all((s == 0.0) | (s == 1.0))

    def loss_fn(m):
        # Push membranes up so spikes actually fire and the surrogate is active.
        spikes = m(u)
        return jnp.mean((spikes - 1.0) ** 2)

    grads = nnx.grad(loss_fn)(mem)
    leaves = jax.tree_util.tree_leaves_with_path(grads)
    norms = {}
    for path, leaf in leaves:
        name = "/".join(str(getattr(p, "key", p)) for p in path)
        norms[name] = float(jnp.linalg.norm(leaf))

    # Gradients must reach the router, the slot dynamics (leak), and the write.
    for component in ("router", "raw_beta", "write"):
        matched = [n for n, v in norms.items() if component in n and v > 0.0]
        assert matched, f"no non-zero gradient reached {component}: {list(norms)}"


def test_spiking_dense_router_reduces_to_psu_lif_bank():
    """With r_t all-ones every slot is an independent, always-written PSU_LIF
    leaky integrator ``V <- beta*V + U_t`` followed by a surrogate spike."""
    rngs = nnx.Rngs(3)
    mem = raven.SpikingSlotMemory(d_model=6, n_slots=5, d_slot=4, rngs=rngs)
    T, B = 9, 3
    u = jax.random.normal(jax.random.PRNGKey(4), (T, B, 6))

    ones = jnp.ones((T, B, mem.n_slots))
    s_dense = mem._run(u, ones)

    # Independent reference: a plain bank of reset-free spiking leaky integrators.
    U = mem.write(u).reshape(T, B, mem.n_slots, mem.d_slot)
    beta = mem.beta  # (M, d_slot)
    v = mem.initial_state(B)
    outs = []
    for t in range(T):
        v = beta[None] * v + U[t]  # r == 1 everywhere: pure PSU_LIF membrane
        outs.append(mem.spike(v - mem.threshold))
    s_ref = jnp.stack(outs, axis=0)

    assert jnp.array_equal(s_dense, s_ref)
