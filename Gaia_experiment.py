from difflib import SequenceMatcher
from typing import Sequence, Callable
from flax import linen as nn
import jax.numpy as jnp
import jax
from annoy import AnnoyIndex
import numpy as np
import h5py
from sklearn.preprocessing import QuantileTransformer

batch_size = 1000
n_epoch = 100

data = h5py.File("/mnt/home/apricewhelan/projects/gaia-scratch/data/gaiadr3-apogee-bprp-Xy.hdf5","r")

X_ = data['X'][:]
Y_ = data['y'][:]
clean_index = np.where(np.invert(np.isnan(Y_[:,2])))[0]
X_ = X_[clean_index]
Y_ = Y_[clean_index]

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
x_scaler = QuantileTransformer()
Y_scaler = QuantileTransformer()
x_scaler.fit(X_)
Y_scaler.fit(Y_)

x_scale = x_scaler.transform(X_)
y_scale = Y_scaler.transform(Y_)

x_index = x_scale[:int(len(x_scale)*0.6)]
x_train = x_scale[int(len(x_scale)*0.6):int(len(x_scale)*0.8)]
y_train = y_scale[int(len(y_scale)*0.6):int(len(y_scale)*0.8)]
x_test = x_scale[int(len(x_scale)*0.8):]
y_test = y_scale[int(len(y_scale)*0.8):]

f = 23  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'angular')

for i in range(len(x_index)):
    if i % 1000 == 0:
        print(i)
    t.add_item(i, x_index[i])

t.build(10) # 10 trees
t.save('test.ann')

key = jax.random.PRNGKey(1071)

train_index = []
for i in range(len(x_train)):
    train_index.append(t.get_nns_by_vector(x_train[i], 9)[1:])

train_index = jnp.stack(np.array(train_index))

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
    
    def __call__(self, index):
        x = self.embbeding(index)
        x = x.reshape(-1, self.embeddingQuerySize*self.embeddingFeatures)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


model = AnnoyMLP(x_index.shape[0], 8, 2, [64, 64, 1])
params = model.init(key, jnp.zeros((1,8), dtype=int))


