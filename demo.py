# %%
import torch
import pax

# %% You can use JAX's pytrees in PyTorch
tree = {
    "a": [torch.tensor(3.0), torch.tensor(4.0)],
    "c": 4
}
pax.tree_map(lambda x: x*2, tree)


# %% You can differentiate
f = lambda x: x**2
pax.grad(f)(2.0)  # tensor(4.0)


# %% You can use any pytree
f = lambda x: x["a"] * x["b"]
x = {"a": 2.0, "b": -1.5}
pax.value_and_grad(f)(x)  # (tensor(-3.), {'a': tensor(-1.5000), 'b': tensor(2.)})


# %% Higher-order derivative
f = lambda x: 1/6 * x**3
pax.grad(f)(2.0)  # tensor(2.)
pax.grad(pax.grad(f))(3.0) # tensor(3.)


# %% A small utility to turn modules functional
net = torch.nn.Sequential(
    torch.nn.Linear(10, 2),
    torch.nn.BatchNorm1d(2),
    torch.nn.Linear(2, 1),
)
forward = pax.functional(net)
params, buffers = net.parameters(), net.buffers()
data = torch.zeros(2, 10)
out, buffers = forward(params, data, buffers=buffers, is_training=True)

print(out, buffers)

# %% A small utility to turn optimizers functional


#%%
import inspect


def wrap(f):
    def call_f(v):
        frame = inspect.stack()
        for line in frame:
            if line.function == "<module>":
                break
            print(line.function, line.filename, line.lineno, line.index)
        return f(v)
    return call_f

@wrap
def f(x):
    return f"hoi {x}"
# %%
wrap(f)(3)
# %%
