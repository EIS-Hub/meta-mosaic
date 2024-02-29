import jax
import jax.numpy as jnp

# dropout function
def dropout(rng, x, rate):
    keep = jax.random.bernoulli(rng, 1 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0)

# LIF neuron model with corresponding JVP
@jax.custom_jvp
def gr_than(x, thr):
    return (x > thr).astype(jnp.float32)
 
@gr_than.defjvp
def gr_jvp(primals, tangents):
    x, thr = primals
    x_dot, y_dot = tangents
    primal_out = gr_than(x, thr)
    # tangent_out = x_dot * 1 / (jnp.absolute(x-thr)+1)**2
    tangent_out = (x_dot / (10 * jnp.absolute(x - thr) + 1) ** 2)
    return primal_out, tangent_out

def lif_forward(state, x):
    ''' Leaky Integrate and Fire (LIF) neuron model
    '''
    inp_weight, rec_weight, bias, out_weight = state[0]     # Static weights
    thr_rec, thr_out, alpha, kappa = state[1]         # Static neuron states
    v, z, vo, zo = state[2]                           # Dynamic neuron states

    v  = alpha * v + jnp.matmul(x, inp_weight) + jnp.matmul(z, rec_weight) + bias - z * thr_rec
    z = gr_than(v, thr_rec)

    vo = kappa * vo + jnp.matmul(z, out_weight)
    # zo = gr_than(vo, thr_out)

    return [[inp_weight, rec_weight, bias, out_weight], [thr_rec, thr_out, alpha, kappa], [v, z, vo, zo]], [z, vo]