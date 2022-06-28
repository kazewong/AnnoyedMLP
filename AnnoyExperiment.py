from annoy import AnnoyIndex
import numpy as np

f = 23  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'angular')

data = np.random.uniform(size=(1000,23))
for i in range(1000):
    t.add_item(i, data[i])

t.build(10) # 10 trees
t.save('test.ann')
