"""
Let us study how probabilistic modeling can estimate uncertainty in parameters of linear regression
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints



def lin_reg_model(X, Y, Z):
    a = pyro.sample("a", dist.Normal(0.0, 5.0))
    b = pyro.sample("b", dist.Normal(0.0, 5.0))
    c = pyro.sample("c", dist.Normal(0.0, 3.0))
    sigma = pyro.sample("sigma", dist.Uniform(0., 50.))
    mean = a * X + b * Y + c

    with pyro.plate("data", len(X)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=Z)

def custom_guide(X, Y, Z=None):
    sigma_loc = pyro.param('sigma_loc', lambda: torch.tensor(0.))
    weights_loc = pyro.param('weights_loc', lambda: torch.randn(3))
    weights_scale = pyro.param('weights_scale', lambda: torch.ones(3),
                               constraint=constraints.positive)
    # sigma_scale = pyro.param("sigma_scale", lambda: torch.tensor(1.5), constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(weights_loc[0], weights_scale[0]))
    b = pyro.sample("b", dist.Normal(weights_loc[1], weights_scale[1]))
    c = pyro.sample("c", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.LogNormal(sigma_loc, torch.tensor(0.2)))  # fixed scale for simplicity
    return {"a": a, "b": b, "c": c, "sigma": sigma}




if __name__ == "__main__":
    X = np.random.uniform(0.0, 10.0, size=200)
    Y = np.random.uniform(0.0, 15.0, size=200)
    T = np.random.uniform(-0.05, 0.05, size=200)
    Z = 2.0 * X + 3.0 * Y + 7.0

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    Z = torch.from_numpy(Z)

    pyro.clear_param_store()

    adam = pyro.optim.Adam({"lr": 0.0005})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(lin_reg_model, custom_guide, adam, elbo)

    losses = []
    for step in range(50000):  # Consider running for more steps.
        loss = svi.step(X, Y, Z)
        losses.append(loss)
        if step % 100 == 0:
            h1 = pyro.param("weights_loc").data.cpu().numpy()
            h2 = pyro.param("weights_scale").data.cpu().numpy()
            print(f"Step {step} Elbo loss: {loss} weights_loc: {h1}, weights_scale: {h2}")

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.grid()
    plt.show()



