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
forward = pax.functional_module(net)
params, buffers = net.parameters(), net.buffers()
data = torch.zeros(2, 10)
out, buffers = forward(params, data, buffers=buffers, is_training=True)

print(out, buffers)

# %% Basic SGD example
import torch
import pax

f = lambda x: x**2
df_dx = pax.grad(f)

x = torch.tensor(2.0)  # initialization
for step in range(20):
    print(x, f(x))
    x = x - 0.1 * df_dx(x)


# %% A small utility to turn optimizers functional
optimizer = pax.functional_optimizer(torch.optim.Adam, lr=1e-3)

f = lambda x: x**2
df_dx = pax.grad(f)
params = torch.tensor(3.)
opt_state = optimizer.init(params)

for step in range(10):
    params, opt_state = optimizer.step(params, df_dx(params), opt_state)
    print(params.item())

# %% It also works with schedulers
import functools

scheduler = functools.partial(
    torch.optim.lr_scheduler.LambdaLR, 
    lr_lambda=lambda step: 1/(step+1)
)

optimizer = pax.functional_optimizer(torch.optim.SGD, lr=.1, scheduler_class=scheduler)

f = lambda x: x**2
df_dx = pax.grad(f)
params = torch.tensor(3.)
opt_state = optimizer.init(params)

for step in range(10):
    params, opt_state = optimizer.step(params, df_dx(params), opt_state)
    opt_state = optimizer.scheduler_step(opt_state)
    print(params.item())
