from difflib import SequenceMatcher
from typing import Sequence, Callable
from flax import linen as nn
import jax.numpy as jnp
import jax
from annoy import AnnoyIndex

key = jax.random.PRNGKey(1071)
data = jax.random.randint(key, shape=(100,8), minval = 0, maxval = 99)

class AnnoyMLP(nn.Module):

    embeddingSize: int
    embeddingQuerySize: int
    embeddingFeatures: int
    mlpFeatures: Sequence[int]
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 0.01
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.embbeding = nn.Embed(self.embeddingSize, self.embeddingFeatures)
        self.layers = [nn.Dense(self.embeddingQuerySize*self.embeddingFeatures),
                        *[nn.Dense(feat, use_bias=self.use_bias, kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"))
                         for feat in self.mlpFeatures]]
    
    def __call__(self, x):
        x = self.embbeding(x)
        x = x.reshape(-1, self.embeddingQuerySize*self.embeddingFeatures)
        print(x.shape)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

model = AnnoyMLP(100, 8, 2, [64, 64, 1])
params = model.init(key, jnp.zeros((1,8), dtype=int))

def loss(x_batched):
    pred = model.apply(params, x_batched)