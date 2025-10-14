# Flow Gym

<div align="center">
  <img src="docs/_static/toy_ode.gif" width="30%" />
  <img src="docs/_static/toy_sde.gif" width="30%" />
  <img src="docs/_static/sd2.gif" width="30%" />
</div>

<p align="center">
<a href="https://github.com/cristianpjensen/flow-gym/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/cristianpjensen/flow_gym"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
<a href="https://github.com/cristianpjensen/flow-gym/actions/"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/cristianpjensen/flow_gym/test.yml?branch=master&logo=github-actions"></a>
</p>

Flow Gym is a library for adapting any base model with any reward function using Value Matching. You only need to specify your data type, model, and reward.

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

## Defining your own environment

In order to apply Value Matching to your own model on your own data type (e.g., molecules, images), you must first define a type that implements the `DataProtocol` (in `flow_gym/types.py`). This includes various arithmetic operations, factory methods, and gradient methods. Then, you need to make sure that your base model outputs this type, usually through a wrapper. Furthermore, you need to make sure that your base model inherits from `BaseModel[YourDataType]`, which means that you must specify the forward pass, how to sample from $p_0$, preprocessing $x_0$ and keyword arguments (e.g., to encode prompts), and postprocessing $x_1$ (e.g., to convert from latent space to pixel space in latent diffusion models). Lastly, you need to implement the reward that you are interested in by inheriting from `Reward[YourDataType]`.

Once all those steps are done, you can either register the base model and reward, or you can use the constructors. We recommend registering and using `flow_gym.make`. This is done as follows:
```python
from flow_gym import base_model_registry, reward_registry, BaseModel

@base_model_registry.register("your_data_type/your_base_model")
class YourBaseModel(BaseModel[YourDataType]):
    ...

@reward_registry.register("your_data_type/your_reward")
class YourReward(BaseModel[YourDataType]):
    ...

env = flow_gym.make("your_data_type/your_base_model", "your_data_type/your_reward", num_steps, reward_scale, device, base_model_kwargs, reward_kwargs)
```
The `data_type`/`name` notation makes it such that only base models and reward with the same `data_type` can be combined.
