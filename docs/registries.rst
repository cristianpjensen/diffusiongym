Using the Registries
====================

When deining a base model and a reward function, it is recommended to register them so that they can be easily combined using ``flowgym.make``.
You can register them as follows:

.. code-block:: python
   :emphasize-lines: 1,3,7

   from flowgym import base_model_registry, reward_registry

   @base_model_registry.register("your_data_type/your_base_model")
   class YourBaseModel(BaseModel[YourDataType]):
      ...

   @reward_registry.register("your_data_type/your_reward")
   class YourReward(BaseModel[YourDataType]):
      ...

The ``data_type``/``name`` notation makes it such that only base models and reward with the same
``data_type`` can be combined. Now, you can create the environment using:

.. code-block:: python
   :emphasize-removed: 1,2,3,4,5
   :emphasize-added: 7,8,9,10,11,12,13

   from flowgym import VelocityEnvironment

   base_model = YourBaseModel(...)
   reward = YourReward(...)
   env = VelocityEnvironment(base_model, reward, discretization_steps=100, reward_scale=100)

   env = flowgym.make(
      "your_data_type/your_base_model",
      "your_data_type/your_reward",
      discretization_steps=100,
      reward_scale=100,
      device=device,
   )

This is very useful for training scripts where you want to try many different combinations of base
models and rewards without changing the code. Additionaly you can pass keyword arguments to the base
model and reward constructors through ``base_model_kwargs`` and ``reward_kwargs`` in
``flowgym.make``.
