# Get Started

## Installation

The base package only contains toy one-dimensional base models and rewards. If you also want access to text-to-image models or molecule models, you have to specify this.
```bash
pip install --upgrade flow_gym
pip install --upgrade flow_gym[images]
pip install --upgrade flow_gym[molecules]
```

## Quickstart

```python
import flow_gym
from flow_gym.methods import value_matching

env = flow_gym.make("images/cifar", "images/compression", discretization_steps=100, reward_scale=100)
samples = env.sample(64)  # sampling from base model

value_network = CNN(...)
value_matching(value_network, env)
samples = env.sample(64)  # sampling from adapted model
```
