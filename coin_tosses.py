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
from scipy.stats import beta


def heads_model(X):
    # apriori distribution for p is uniform
    p = pyro.sample("p", dist.Uniform(0, 1))

    with pyro.plate("data", len(X)):
        return pyro.sample("obs", dist.Bernoulli(probs=p), obs=X)

def custom_guide(X):
    a1_loc = pyro.param("a1_loc", lambda: torch.tensor(0.1), constraint=constraints.positive)
    a2_loc = pyro.param("a2_loc", lambda: torch.tensor(0.9), constraint=constraints.less_than(1.0))
    p = pyro.sample("p", dist.Uniform(a1_loc, a2_loc))
    return {"p": p}


def better_guide(X):
    alpha = pyro.param("alpha", torch.tensor(3.0), constraint=constraints.positive)
    beta = pyro.param("beta", torch.tensor(3.0), constraint=constraints.positive)

    p = pyro.sample("p", dist.Beta(alpha, beta))
    return {"p": p}


if __name__ == "__main__":
    p = 0.73
    n = 400

    X = torch.from_numpy(np.random.binomial(1, p, size=n)).float()

    # guide_to_use = pyro.infer.autoguide.AutoNormal(heads_model)
    # guide_to_use = custom_guide
    guide_to_use = better_guide

    pyro.clear_param_store()

    adam = pyro.optim.Adam({"lr": 1.e-2})
    elbo = pyro.infer.Trace_ELBO()
    # elbo = pyro.infer.TraceMeanField_ELBO()
    svi = pyro.infer.SVI(heads_model, guide_to_use, adam, elbo)

    losses = []
    for step in range(15000):  # Consider running for more steps.
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

    num_samples = 5000
    posterior_samples = []

    for _ in range(num_samples):
        sample = guide_to_use(X)
        posterior_samples.append(sample["p"].item())

    posterior_samples = np.array(posterior_samples)

    x_sum = X.sum().item()
    alpha_true = 1 + x_sum
    beta_true = 1 + n - x_sum

    x_grid = np.linspace(0, 1, 200)
    true_pdf = beta.pdf(x_grid, alpha_true, beta_true)



    plt.figure(figsize=(6, 4))
    plt.hist(posterior_samples, bins=30, density=True, alpha=0.6, label="Posterior samples")

    # истинное значение
    plt.axvline(p, color="red", linestyle="--", label=f"True p = {p}")
    plt.plot(x_grid, true_pdf, label="True posterior (Beta)", color="black")
    plt.xlabel("p")
    plt.ylabel("Density")
    plt.title("Posterior distribution of p")
    plt.legend()
    plt.grid()
    plt.show()



