Setting RL/SOC Policies
=======================

Policies are defined as base models, and are set through the ``Environment.policy`` or
``Environment.control_policy`` properties. If you want to initialize the policy as the pre-trained base model, you can do so by copying the base model:

.. code-block:: python

   from copy import deepcopy

   policy = deepcopy(env.base_model)
   env.policy = policy

Now ``Environment.sample`` will use ``policy`` for sampling instead of the pre-trained base model,
and the running costs will be computed based on the difference between the policy and the pre-trained
base model. The pre-trained model can still be accessed through ``env.base_model``.

If you have a network that represents the control :math:`u(\mathbf{x}_t, t)` directly, you can set it
as the control policy:

.. code-block:: python

   from flowgym import ControlPolicy

   control_policy = ControlPolicy(your_control_network)
   env.control_policy = control_policy

Then the drift is computing as :math:`b(\mathbf{x}_t, t) + \sigma(t) u(\mathbf{x}_t, t)` during
sampling, and the running cost can be computed more efficiently.
