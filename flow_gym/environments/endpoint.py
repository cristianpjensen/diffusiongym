r"""Environment with tensor samples and base model predicts the endpoint :math:`x_1(x, t)`."""

from typing import Any

import torch

from flow_gym.types import DataType

from .base import BaseEnvironment


class EndpointEnvironment(BaseEnvironment[DataType]):
    r"""Environment with tensor samples and base model predicts the endpoint :math:`x_1(x, t)`.

    Parameters
    ----------
    base_model : BaseModel
        The base generative model used in the environment.

    reward : Reward
        The reward function used to compute the final reward.

    discretization_steps : int
        The number of discretization steps to use when sampling trajectories.
    """

    def drift(
        self,
        x: DataType,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[DataType, torch.Tensor]:
        """Compute the drift term of the environment's dynamics.

        Parameters
        ----------
        x : DataType
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        **kwargs : dict
            Additional keyword arguments to pass to the base model (e.g. text embedding or class
            label).

        Returns
        -------
        drift : DataType
            The drift term at state x and time t.

        running_cost : torch.Tensor, shape (n,)
            Running cost :math:`L(x_t, t)` of the policy for the given (state, timestep)-pair.
        """
        alpha = self.scheduler.alpha(x, t)
        beta = self.scheduler.beta(x, t)
        kappa = self.scheduler.kappa(x, t)
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)
        sigma_eta = 0.5 * sigma * sigma + eta
        sigma_ft = self.memoryless_schedule(x, t)

        action = self.policy(x, t, **kwargs)

        a = kappa - sigma_eta / (beta * beta)
        b = sigma_eta * alpha / (beta * beta)

        control = x.zeros_like()
        if self.is_policy_set:
            action_base = self.base_model.forward(x, t, **kwargs)
            control = (b / sigma_ft) * (action - action_base)

        if self.control_policy is not None:
            control_add = self.control_policy(x, t, **kwargs)
            control += control_add
            action += (sigma_ft / b) * control_add

        return a * x + b * action, 0.5 * (control * control).batch_sum()
