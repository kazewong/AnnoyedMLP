from difflib import SequenceMatcher
from typing import Sequence, Callable
from flax import linen as nn
from flax.training import train_state
import jax.numpy as jnp
import jax
from annoy import AnnoyIndex
import numpy as np
import h5py
from sklearn.preprocessing import QuantileTransformer
import optax
import copy

batch_size = 1000
n_epoch = 100

data = h5py.File("/mnt/home/apricewhelan/projects/gaia-scratch/data/gaiadr3-apogee-bprp-Xy.hdf5","r")

X_ = data['X'][:]
Y_ = data['y'][:,:1]
# clean_index = np.where(np.invert(np.isnan(Y_[:,2])))[0]
# X_ = X_[clean_index]
# Y_ = Y_[clean_index]

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
x_scaler = QuantileTransformer()
Y_scaler = QuantileTransformer()
x_scaler.fit(X_)
Y_scaler.fit(Y_)

x_scale = X_#x_scaler.transform(X_)
y_scale = Y_#Y_scaler.transform(Y_)

x_index = x_scale[:int(len(x_scale)*0.2)]
y_index = y_scale[:int(len(y_scale)*0.2)]
x_train = x_scale[int(len(x_scale)*0.2):int(len(x_scale)*0.8)]
y_train = y_scale[int(len(y_scale)*0.2):int(len(y_scale)*0.8)]
x_test = x_scale[int(len(x_scale)*0.8):]
y_test = y_scale[int(len(y_scale)*0.8):]

f = 23  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'angular')

for i in range(len(x_index)):
    t.add_item(i, x_index[i])

t.build(10) # 10 trees
t.save('test.ann')


train_index = []

for i in range(len(x_train)):
    train_index.append(t.get_nns_by_vector(x_train[i], 9)[1:])
train_index = jnp.stack(np.array(train_index))

test_index = []
for i in range(len(x_test)):
    test_index.append(t.get_nns_by_vector(x_test[i], 9)[1:])
test_index = jnp.stack(np.array(test_index))

class AnnoyMLP(nn.Module):

    embeddingSize: int
    embeddingQuerySize: int
    embeddingFeatures: int
    mlpFeatures: Sequence[int]
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 0.1
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.embedding = nn.Embed(self.embeddingSize, self.embeddingFeatures)
        self.layers = [nn.Dense(self.embeddingQuerySize*self.embeddingFeatures),
                        *[nn.Dense(feat, use_bias=self.use_bias, kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"))
                         for feat in self.mlpFeatures],nn.Dense(self.embeddingQuerySize)]
    
    def __call__(self, index):
        x = self.embedding(index)
        x = x.reshape(-1, self.embeddingQuerySize*self.embeddingFeatures)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        output =  self.layers[-1](x)
        return jnp.mean(output,axis=-1,keepdims=True)

class EmbeddingMean(nn.Module):

    embeddingSize: int
    embeddingFeatures: int

    def setup(self):
        self.embedding = nn.Embed(self.embeddingSize, self.embeddingFeatures)
    
    def __call__(self, index):
        x = self.embedding(index)[:,:,0]
        return jnp.mean(x,axis=-1,keepdims=True)


model = AnnoyMLP(x_index.shape[0], 8, 1, [32, 32])
model_baseline = EmbeddingMean(x_index.shape[0], 1)


key = jax.random.PRNGKey(1071)
rng, init_rng = jax.random.split(key, 2)
learning_rate = 10
momentum = 0.9

params = model.init(rng, jnp.zeros((100,8), dtype=int))
# params = model_baseline.init(rng, jnp.zeros((100,8), dtype=int))

tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)


@jax.jit
def train_step(params, opt_state, x, y):
    def loss(params):
        # pred_y = model_baseline.apply(params, x)
        pred_y = model.apply(params, x)
        return jnp.mean(jnp.square(pred_y - y))
    grad_fn = jax.value_and_grad(loss)
    value, grad = grad_fn(params)
    updates, opt_state = tx.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

@jax.jit
def eval_step(params, x, y):
    # pred_y = model_baseline.apply(params, x)
    pred_y = model.apply(params, x)
    return jnp.mean(jnp.square(pred_y - y))


num_epochs = 30000

loss = 1e10
for epoch in range(1, num_epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    params, opt_state = train_step(params, opt_state, train_index, y_train)
    loss_local = eval_step(params, test_index, y_test)
    if epoch % 100 == 0:
        print('Epoch %d' % epoch)
        print('Loss: %.3f' % loss_local)
    if loss_local < loss:
        loss = loss_local
        save_params = copy.copy(params)
        save_state = copy.copy(opt_state)
