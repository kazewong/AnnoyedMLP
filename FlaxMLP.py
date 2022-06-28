from typing import Sequence, Callable
from flax import linen as nn
import jax.numpy as jnp
import jax

key = jax.random.PRNGKey(1071)
data = jax.random.normal(key, shape=(100,10))

model = nn.Embed(10,5)

params = model.init(key, jnp.zeros(1))['params']

class AnnoyMLP(nn.Module):
