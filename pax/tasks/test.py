#%%
%load_ext autoreload
%autoreload 2
#%%
import torch
import pax
from pax import tasks
# %%
print(tasks.list())

# %%
task = tasks.get("deepobs.quadratic_deep")()
print(task.name)
grad = pax.grad(task.loss, has_aux=True)

# %%
sgd_step = lambda param, grad: param - learning_rate * grad

#%%
learning_rate = 0.01
params, buffers = task.init()
for epoch in range(2):
    for batch in task.train.iterator(batch_size=10, num_workers=2):
        minibatch_grad, buffers = grad(params, batch, buffers=buffers)
        params = pax.tree_map(sgd_step, params, minibatch_grad)
    print("test", task.evaluate(params, buffers=buffers))  # returns dict with cross_entropy and accuracy
    print("train", task.evaluate(params, task.train, buffers=buffers))  # for training stats

# %%
