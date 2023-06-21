class ActivityRegularization(hk.Module):
    def __init__(self, name="ActReg"):
        super().__init__(name=name)
        
    def __call__(self, spikes):
        spike_count = hk.get_state("spike_count", [spikes.shape[-1]], init=jnp.zeros)
        hk.set_state("spike_count", spike_count + jnp.mean(spikes, axis=0))
        return spikes


class Heaviside: # not sure where to ultimately put this
    def __init__(self, scale_factor=25):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), ()
            
        # Straight Through Estimator
        def f_bwd(U, grad):
            return ( grad*U ) 
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U)


# Surrogate functions
class Arctan: # not sure where to ultimately put this
    def __init__(self, scale_factor=2):
        self.a = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), ()
            
        # Straight Through Estimator
        def f_bwd(U, grad):
            return ( (1 / (jnp.pi * (1+(jnp.pi*U*a/2)**2))) * grad ) 
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U) 
    
class Sigmoid:
    def __init__(self, scale_factor=25):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), U
            
        # accepts context, primal val
        def f_bwd(U, grad): # is spk needed at all???
            return ((self.k*jnp.exp(-self.k*U) / (jnp.exp(-self.k*U)+1)**2) * grad)
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, V, T):
        return self.f(V, T)

class SuperSpike:
    def __init__(self, scale_factor=25):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), U
            
        # accepts context, primal val
        def f_bwd(U, grad):
            return ((1 / (1+self.k*jnp.abs(U))**2) * grad, )
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U)
    
class AdaSpike:
    def __init__(self,  warmup_steps, growth_rate=0.5):
        self.k = 1
        self.gr = growth_rate
        
        @jax.custom_vjp
        def f(U, k): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U, k):
            return f(U), (U, k)
            
        # accepts context, primal val
        # not sure if k actually changes or it gets jit'ed and stays static..
        def f_bwd(context, grad):
            U, k = context
            return ((1 / (1+k*jnp.abs(U))) * grad, )
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        self.k += self.growth_rate
        return self.f(U, self.k)
        
class Boxcar:
    def __init__(self, scale_factor=1):
        self.k = scale_factor
        
        @jax.custom_vjp
        def f(U): # primal function
            return (U>0).astype(jnp.float16)
        
        # returns value, grad context
        def f_fwd(U):
            return f(U), U
            
        # Needs fixed.
        def f_bwd(U, grad):
            if jnp.abs(U) <= 0.5:
                return ( 0.5 * grad )
            return ( 0 ) 
            
        f.defvjp(f_fwd, f_bwd)
        self.f = f
        
    def __call__(self, U):
        return self.f(U)
