The Math
========

.. rst-class:: lead

   *flowgym* is built for reward adaptation of flow and diffusion models. Here, we provide a brief
   overview of the mathematical framework used in the library.

Flow models
-----------

The idea of diffusion and flow matching models (collectively referred to as flow models) is to
construct a process with the same time marginals as the reference flow. As such, simulating the
process from :math:`t=0` to :math:`t=1` transforms samples from :math:`p_0` to samples from the
target distribution.

Given an initial distribution :math:`\mathbf{x}_0 \sim p_0` and samples :math:`\mathbf{x}_1` from a
target distribution, the reference flow is defined as:

.. math::
   :label: reference_flow

   \mathbf{x}_t = \alpha_t \mathbf{x}_0 + \beta_t \mathbf{x}_1

where :math:`\alpha_t` and :math:`\beta_t` are scalar functions of time :math:`t` satisfying
:math:`\alpha_0 = \beta_1 = 0` and :math:`\alpha_1 = \beta_0 = 1`.

.. note::

   In *flowgym*, :math:`\alpha_t` and :math:`\beta_t` are defined as subclasses of the
   ``Scheduler`` abstract class.

Generally, flow matching trains a velocity field :math:`v(\mathbf{x}_t, t)` to match the time
derivative of the reference flow. Then, the process is defined as the ordinary differential equation
(ODE):

.. math::
   :label: ode

   \mathrm{d} \mathbf{x}_t = v(\mathbf{x}_t, t)\,\mathrm{d}t

However, we can also choose to sample from a family of stochastic differential equations (SDEs) that
have the same time marginals:

.. math::
   :label: sde_family

   \mathrm{d} \mathbf{x}_t = \left( v(\mathbf{x}_t, t) + \frac{\sigma^2(t)}{2\beta_t \left( \frac{\dot{\alpha}_t}{\alpha_t} \beta_t - \dot{\beta}_t \right)} \left( v(\mathbf{x}_t, t) - \frac{\dot{\alpha}_t}{\alpha_t} \mathbf{x}_t \right) \right)\,\mathrm{d}t + \sigma(t)\,\mathrm{d}B_t

where :math:`\sigma(t)` is an arbitrary diffusion coefficient and :math:`B_t` is standard Brownian motion.

From now on, we will view the drift term in :eq:`sde_family` as a constant defined through the base model:

.. math::
   :label: sde

   \mathrm{d} \mathbf{x}_t = b(\mathbf{x}_t, t)\,\mathrm{d}t + \sigma(t)\,\mathrm{d}B_t

where :math:`b(\mathbf{x}_t, t)` is the drift term. The ``Environment`` classes in *flowgym*
implements Euler-Maruyama sampling of this SDE, where the drift is defined through a ``BaseModel``.

.. note::

   The base model does not have to output the velocity field :math:`v(\mathbf{x}_t, t)`. It can also
   output the marginal noise :math:`\epsilon(\mathbf{x}_t, t)` as in diffusion models, the endpoint
   :math:`\hat{\mathbf{x}}_1(\mathbf{x}_t, t)`, or the score :math:`\nabla_x \log p_t(\mathbf{x}_t)`. These
   are all equivalent up to a re-scaling [#fmgc]_. You only need to make sure to choose the correct
   environment: ``VelocityEnvironment``, ``EpsilonEnvironment``, ``EndpointEnvironment``, or
   ``ScoreEnvironment``.

Reward adaptation
-----------------

In order to adapt the base model to a task, a reward function :math:`r` is introduced that is
evaluated at the end of the generative process, i.e., :math:`t=1`. The idea of reward adaptation
is to adapt the drift term in :eq:`sde` such that samples :math:`\mathbf{x}_1` have high reward.

.. note::

   In *flowgym*, reward functions are implemented as subclasses of the ``Reward`` abstract class.

A common objective is KL-regularized reward maximization:

.. math::
   :label: kl_reward_objective

   \pi^{\star} \in \arg\max \mathbb{E}_{p_1^{\pi}} \left[ r(\mathbf{x}_1) \right] - D_{\mathrm{KL}} \left( p_1^{\pi} \;\middle|\; p_1 \right)

Many works [#soc]_ propose an equivalent SOC formulation where a control term :math:`u(\mathbf{x}_t, t)` is added to the
drift:

.. math::
   :label: controlled_sde

   \mathrm{d} \mathbf{x}_t = \left( b(\mathbf{x}_t, t) + \sigma(t) u(\mathbf{x}_t, t) \right)\,\mathrm{d}t + \sigma(t)\,\mathrm{d}B_t

And the objective is to minimize the cost functional at every state :math:`(\mathbf{x}_t, t)`:

.. math::
   :label: soc_objective

   J(u; \mathbf{x}_t, t) = \mathbb{E}_{p^u} \left[ \frac{1}{2} \int_t^1 \| u(\mathbf{x}_s, s) \|^2 \,\mathrm{d}s - r(\mathbf{x}_1) \;\middle|\; \mathbf{x}_t \right]

.. note::

   Given a policy, the ``Environment.sample`` method simulates the controlled SDE and returns the
   rewards and cost functionals over the sampled trajectory. It also returns other data such as
   drifts and noises that may be useful for some algorithms.

Fine-tuning schemes define the control :math:`u(\mathbf{x}_t, t)` through the controlled and uncontrolled base models:

.. math::
   :label: control_regression

   u(\mathbf{x}_t, t) = \sigma^{-1}(t) \left( b^{\star}(\mathbf{x}_t, t) - b(\mathbf{x}_t, t) \right)

where :math:`b^{\star}(\mathbf{x}_t, t)` is the drift of the controlled process.

Alternative methods instead learn an auxiliary value function :math:`V(\mathbf{x}_t, t)` defined as the optimal cost-to-go:

.. math::
   :label: value_function

   V(\mathbf{x}_t, t) = \inf_{u} J(u; \mathbf{x}_t, t)

The optimal control can then be obtained from the value function:

.. math::
   :label: optimal_control

   u^{\star}(\mathbf{x}_t, t) = -\sigma^\top(t) \nabla_x V(\mathbf{x}_t, t)

This method is more flexible in terms of resource requirements and does not require the reward
function to be differentiable.

.. note::

   To facilitate this, *flowgym* provides the ``ValuePolicy`` class that derives the optimal control for the
   value function approximator. This can be used with the ``Environment`` class by setting the
   ``control_policy`` property.

.. rubric:: Footnotes

.. [#fmgc] See Table 1 of *Lipman, Yaron, et al. "Flow matching guide and code." arXiv preprint arXiv:2412.06264 (2024)*.

.. [#soc] For example, see *Domingo-Enrich, Carles, et al. "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control" (2025)*.
