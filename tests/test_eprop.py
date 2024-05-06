from spyx.axn import abs_linear
from spyx.nn import RecurrentLIFLight, LeakyLinear, LI
import jax.numpy as jnp
import jax
import numpy as np

import haiku as hk


def test_axn_abs_linear():
    """
    Test the abs_linear surrogate gradient function.
    """

    # Test the abs_linear surrogate gradient function.
    x = jnp.linspace(-10, 10, 100, dtype=jnp.float32)
    f = abs_linear(dampening_factor=0.3)

    y = f(x)
    y_grad = jax.jacrev(f)(x).diagonal()

    y_true = jnp.greater(x, 0.).astype(jnp.float32)
    y_true_grad = jnp.maximum(0.3*(1 - jnp.abs(x)), 0).astype(jnp.float32)
 
    assert np.allclose(y, y_true, atol=1e-5)
    assert np.allclose(y_grad, y_true_grad, atol=1e-5)

def test_eprop():
    n_in = 3
    n_LIF = 2
    n_ALIF = 2
    n_rec = n_ALIF + n_LIF

    dt = 1  # ms
    tau_v = 20  # ms
    tau_a = 500  # ms
    T = 100  # ms
    f0 = 100  # Hz

    thr = 0.62 
    beta = 0.07 * jnp.concatenate([jnp.zeros(n_LIF), jnp.ones(n_ALIF)])
    dampening_factor = 0.3
    n_ref = 3
    batch_size = 5

    key = jax.random.PRNGKey(2)
    inputs = (jax.random.uniform(key, shape=(1, T, n_in)) < f0 * dt / 1000).astype(float)
    print(inputs.shape, inputs)

    def lsnn(x, state=None, batch_size=1):
        core = hk.DeepRNN([
            hk.Linear(n_rec),
            RecurrentLIFLight(
                n_rec,
                tau=tau_v,
                thr=thr,
                dt=dt,
                dtype=jnp.float32,
                dampening_factor=dampening_factor,
                tau_adaptation=tau_a,
                beta=beta,
                tag='',
                stop_gradients=True,
                w_rec_init=None,
                n_refractory=n_ref,
                rec=True,
            ),
            # LeakyLinear(n_rec, 20, jnp.exp(-dt/tau_v))
            hk.Linear(20, with_bias=False, w_init=hk.initializers.Constant(
                (1-jnp.exp(-dt/tau_v))*jnp.eye(n_rec, 20))),
            LI((20,), jnp.exp(-dt/tau_v))
        ])
        if state is None:
            state = core.initial_state(batch_size)
        spikes, hiddens = core(x, state)
        return spikes, hiddens

    lsnn_hk = hk.without_apply_rng(hk.transform(lsnn))
    # i0 = jnp.stack([inputs[:,0], inputs[:,0], inputs[:,0],inputs[:,0], inputs[:,0]], axis=0)
    # i0 = jnp.zeros((batch_size, n_in))
    i0 = []
    for _ in range(batch_size):
        i0.append(inputs[:,0])
    i0 = jnp.stack(i0, axis=0)
    print(i0.shape)
    params = lsnn_hk.init(rng=key, x=i0, batch_size=batch_size)
    print(params)
    # w_in_copy = [[ 0.7967948 , -0.3821632 , -0.7605332 ,  0.45293623],
    #    [-0.03456055,  0.65856   ,  0.58331513, -0.10983399],
    #    [-0.4869853 ,  1.0580422 ,  0.53946483, -0.00187313]]
    # w_in_copy = jnp.array(w_in_copy)

    state = None
    spikes = []
    V = []
    variations = []
    # if w_rec is not None:
    #     params['RecurrentLIFLight']['w_rec'] = w_rec
    # if w_in is not None:
    #     params['linear']['w'] = w_in
    # if w_out is not None:
    #     params['LeakyLinear']['weights'] = w_out
    for t in range(T):
        it = inputs[:, t]
        it = jnp.expand_dims(it, axis=0)
        outs, state = lsnn_hk.apply(params, it, state, batch_size)
        # print(inputs[:,t], "->", outs)
        spikes.append(outs)

    y_out = jnp.stack([s[0] for s in spikes], axis=0)
    y_target = jax.random.normal(key=key, shape=[T, 1])
    print(y_out.shape, y_target.shape)
    loss = 0.5 * jnp.sum((y_out - y_target) ** 2)
    y_out = jnp.expand_dims(y_out, axis=0)
    y_target = jnp.expand_dims(y_target, axis=0)

    print(loss)
    loss_target = 838.4397

    assert np.allclose(loss, loss_target, atol=1e-5)

    # TODO grad compute eprop and bptt

if __name__ == "__main__":
    # test_axn_abs_linear()
    test_eprop()