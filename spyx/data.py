import jax
import jax.numpy as jnp


# This should be changed to a higher-order function
def shift_augment(max_shift=10, axes=(-1,)):
    """Shift data augmentation tool. Rolls data along specified axes randomly up to a certain amount.

        
    :max_shift: maximum to which values can be shifted.
    :axes: the data axis or axes along which the input will be randomly shifted.
    """

    def _shift(data, rng):
        shift = jax.random.randint(rng, (len(axes),), -max_shift, max_shift)
        return jnp.roll(data, shift, axes)
    
    return jax.jit(_shift)


def shuffler(dataset, batch_size):
    """
    Higher-order-function which builds a shuffle function for a dataset.

    :dataset: jnp.array [# samples, time, channels...]
    :batch_size: desired batch size.
    """
    cutoff = y.shape[0] % batch_size
    data_shape = (-1, batch_size) + obs.shape[1:]

    def _shuffle(dataset, shuffle_rng):
        """
        Given a dataset as a single tensor, shuffle its batches.

        :dataset: tuple of jnp.arrays with shape [# batches, batch size, time, ...] and [# batches, batchsize]
        :shuffle_rng: JAX.random.PRNGKey
        """
        x, y = dataset

        obs = jax.random.permutation(shuffle_rng, x, axis=0)[:-cutoff]
        labels = jax.random.permutation(shuffle_rng, y, axis=0)[:-cutoff]

        obs = jnp.reshape(obs, data_shape)
        labels = jnp.reshape(labels, (-1, batch_size)) # should make batch size a global

        return (obs, labels)

    return jax.jit(_shuffle)




def rate_code(num_steps, max_r=0.75):
    """
    Unrolls input data along axis 1 and converts to rate encoded spikes; the probability of spiking is based on the input value multiplied by a max rate, with each time step being a sample drawn from a Bernoulli distribution.
    Currently Assumes input values have been rescaled to between 0 and 1.
    """

    def _call(data, key):
        data = jnp.array(data, dtype=jnp.float16)
        unrolled_data = jnp.repeat(data, steps, axis=1)
        return jax.random.bernoulli(key, unrolled_data*max_r).astype(jnp.uint8)
    
    return jax.jit(_call)



def angle_code(neuron_count, min_val, max_val):
    """
    Higher-order-function which returns an angle encoding function; given a continuous value, an angle converter generates a one-hot vector corresponding to where the value falls between a specified minimum and maximum.
    To achieve non-linear descritization, apply a function to the continuous value before feeding it into the encoder.

    :neuron_count: The number of output channels for the angle encoder
    :min_val: A lower bound on the continuous input channel
    :max_val: An upper bound on the continuous input channel.
    """
    neurons = jnp.linspace(min_val, max_val, neuron_count)
        
    def _call(obs):
        digital = jnp.digitize(obs, neurons)
        return jax.nn.one_hot(digital, neuron_count)

    return jax.jit(_call)