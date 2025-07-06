"""
Example hidden Markov model obtined from chatGPT
With some corrections (the example 'as it is' did not work)
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDelta


import pyro.poutine as poutine

pyro.clear_param_store()

# Параметры модели
num_steps = 10  # длина наблюдаемой последовательности
num_states = 3  # количество скрытых состояний

# Генерация искусственных данных
true_transition_probs = torch.tensor([
    [0.9, 0.05, 0.05],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
true_emission_locs = torch.tensor([-3.0, 0.0, 3.0])
true_sigma = 0.5


def generate_data():
    z = 0
    zs = [z]
    xs = []

    for t in range(num_steps):
        z = dist.Categorical(true_transition_probs[z]).sample()
        zs.append(z.item())
        x = dist.Normal(true_emission_locs[z], true_sigma).sample()
        xs.append(x)
    return torch.stack(xs), torch.tensor(zs)


data, true_zs = generate_data()


# Модель
def model(data):
    transition_probs = pyro.param("transition_probs", torch.ones(num_states, num_states) / num_states,
                                  constraint=dist.constraints.simplex)
    emission_locs = pyro.param("emission_locs", torch.linspace(-2, 2, num_states))
    sigma = pyro.param("sigma", torch.tensor(1.0), constraint=dist.constraints.positive)

    z_prev = pyro.sample("z_start", dist.Categorical(torch.ones(num_states) / num_states),
                         infer={"enumerate": "parallel"})

    for t in range(len(data)):
        z_t = pyro.sample(f"z_{t}", dist.Categorical(transition_probs[z_prev]),
                          infer={"enumerate": "parallel"})
        pyro.sample(f"x_{t}", dist.Normal(emission_locs[z_t], sigma), obs=data[t])
        z_prev = z_t


# Guide (детерминированный — AutoDelta)
guide = AutoDelta(poutine.block(model, hide=['z_start'] + [f'z_{t}' for t in range(len(data))]))

# Optimizer and inference
optimizer = Adam({"lr": 0.05})
elbo = TraceEnum_ELBO(max_plate_nesting=0)
svi = SVI(model, guide, optimizer, loss=elbo)

# Обучение
for step in range(20000):
    loss = svi.step(data)
    if step % 50 == 0:
        print(f"Step {step}:\tELBO = {loss:.2f}")

# Печать оценённых параметров
print("\nОценённые параметры:")
print("transition_probs:\n", pyro.param("transition_probs").data)
print("emission_locs:\n", pyro.param("emission_locs").data)
print("sigma:\n", pyro.param("sigma").item())