# Defining Your Own Models and Rewards

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
