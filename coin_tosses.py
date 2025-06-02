"""
This is Bayesian coin toss experiment
We generate a sequence of Bernoulli variables and estimate the posterior distribution
of p parameter
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints


def heads_model(X):
    # apriori distribution for p is uniform
    p = pyro.sample("p", dist.Uniform(0, 1))

    with pyro.plate("data", len(X)):
        return pyro.sample("obs", dist.Bernoulli(p), obs=X)

def custom_guide(X):
    a1_loc = pyro.param("a1_loc", lambda: torch.tensor(0.1), constraint=constraints.positive)
    a2_loc = pyro.param("a2_loc", lambda: torch.tensor(0.9), constraint=constraints.less_than(1.0))
    p = pyro.sample("p", dist.Uniform(a1_loc, a2_loc))
    return {"p": p}


if __name__ == "__main__":
    p = 0.73
    n = 400

    X = torch.from_numpy(np.random.binomial(1, p, size=n)).float()

    # guide_to_use = pyro.infer.autoguide.AutoNormal(heads_model)
    guide_to_use = custom_guide

    pyro.clear_param_store()

    adam = pyro.optim.Adam({"lr": 1.e-3})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(heads_model, guide_to_use, adam, elbo)

    losses = []
    for step in range(5000):  # Consider running for more steps.
        loss = svi.step(X)
        losses.append(loss)
        if step % 100 == 0:
            print(f"Step {step} Elbo loss: {loss}")

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.grid()
    plt.show()


