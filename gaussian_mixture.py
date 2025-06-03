import numpy as np
import torch
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from sklearn.mixture import GaussianMixture


def mixture_model(X):
    k = 5

    locs = pyro.param("locs", torch.zeros(k))
    scales = pyro.param("scales", torch.ones(k), constraint=constraints.positive)
    weights = pyro.param("weights", torch.ones(k), constraint=constraints.positive)
    weights_norm = weights / weights.sum()

    with pyro.plate("data", len(X)):
        comp_n = pyro.sample("component", dist.Categorical(weights_norm), infer={"enumerate": "parallel"})
        return pyro.sample("mix", dist.Normal(locs[comp_n], scales[comp_n]), obs=X)

def custom_guide(X):
    pass


if __name__ == "__main__":
    p = 0.73
    n = 400

    d = 1
    k = 5
    _weights = np.array([0.1, 0.1, 0.3, 0.2, 0.3])
    data_gmm = GaussianMixture(n_components=k)
    data_gmm.weights_ = _weights / _weights.sum()
    data_gmm.means_ = np.array([0.1, -0.7, 1.5, -1.4, 2.8]).reshape((-1, 1))
    data_gmm.covariances_ = np.array([1.0, 1.0, 2.0, 3.0, 2.0]).reshape((-1, 1, 1))
    X, _ = data_gmm.sample(n)

    # print(X)

    X = torch.from_numpy(X).squeeze()
    print("X shape:", X.shape)

    guide_to_use = custom_guide

    pyro.clear_param_store()

    adam = pyro.optim.Adam({"lr": 2.e-3})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model=mixture_model, guide=guide_to_use, optim=adam, loss=elbo)

    losses = []
    for step in range(1000000):  # Consider running for more steps.
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



