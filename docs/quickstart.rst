Quickstart
==========

This quickstart guides you through how to define your own data type, base model, and reward function.

Data type
---------

In order to sample a base model and use it for reward adaptation, you need to first define a data type for your modality. This data type must implement the ``FlowProtocol``, which defines:

- ``apply``, which applies a unary operation on the underlying data.
- ``combine``, which applies a binary operation on the underlying data.
- ``aggregate``, which computes a sum over the underlying data per sample in the batch.
- ``collate``, which collates a list of data type instances into a batched instance.
- ``__len__`` and ``__getitem__`` to support indexing.

See ``flowgym/types.py`` for more details on the methods, and see ``flowgym/molecules/types.py`` for an example implementation for molecules using an underlying DGL graph.

Base model
----------

Next, you need to define a base model that operates on this data type. This base model must inherit from ``BaseModel[YourDataType]``, which requires you to implement:

- ``scheduler (Scheduler[YourDataType])``: The :math:`(\alpha_t, \beta_t)` flow matching schedule.
- ``sample_p0``: How to sample from :math:`p_0`.
- ``forward``: The forward pass of the base model, which takes in :math:`(x_t, t)` and outputs one of the following: marginal noise :math:`\epsilon`, velocity :math:`v`, endpoint :math:`x_1`, or score :math:`\nabla_x \log p_t(x_t)`. Make sure to set ``output_type`` accordingly, so ``flowgym.make`` will know what environment to use.
- ``preprocess``: Preprocess procedure to convert :math:`x_0` and any keyword arguments (e.g., to encode prompts).
- ``postprocess``: Postprocess procedure to convert :math:`x_1` to the desired output format (e.g., convert from latent space to pixel space in latent diffusion models).

Reward function
---------------

Finally, you need to define a reward that operates on this data type. This reward must inherit from ``Reward[YourDataType]``, which requires you to implement:

- ``forward``: The forward pass of the reward, which takes in :math:`x_1` and any keyword arguments, and outputs a reward for each sample in the batch.

Defining an environment
-----------------------

Once you have defined a base model and a reward function, they can be combined into an environment.
Assuming you have defined ``YourBaseModel`` to predict the velocity field, you can create the environment as follows:

.. code-block:: python

   from flowgym import VelocityEnvironment

   base_model = YourBaseModel(...)
   reward = YourReward(...)
   env = VelocityEnvironment(
      base_model,
      reward,
      discretization_steps=100,
      reward_scale=100,
   )

Here, ``discretization_steps`` is the number of time steps to simulate the flow SDE, and
``reward_scale`` is a scaling factor applied to the reward to balance reward maximization and staying
close to the base model in reward adaptation algorithms.

Sampling
--------

Now you can sample from the base model using:

.. code-block:: python

   x1, traj, drifts, noises, running_costs, rewards, valids, costs, kwargs = env.sample(n, pbar=True, **kwargs)

Here, ``x1`` is the final sample, ``traj`` are the full trajectories from :math:`t=0` to
:math:`t=1`, ``drifts`` are the drift terms at each time step, ``noises`` are the noise terms at each
time step, ``running_costs`` are the running costs at each time step, ``rewards`` are the rewards
computed by the reward function, ``valids`` are boolean indicators of whether each sample is valid
(if applicable), ``costs`` are the total costs (negative rewards plus running costs), and ``kwargs``
contains any additional information returned by the base model or reward.

This should be all the information you need for an RL or SOC algorithm to optimize for the reward
using the base model as the underlying pre-trained generative model! If not, open an issue on GitHub.
