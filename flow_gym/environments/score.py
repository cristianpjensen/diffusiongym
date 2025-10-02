r"""Environment with tensor samples and base model predict score :math:`\nabla \log p_t(x)`."""

from typing import Any

import torch

from flow_gym.utils import DataType

from .base import BaseEnvironment


class ScoreEnvironment(BaseEnvironment[DataType]):
    r"""Environment with tensor samples and base model predict score :math:`\nabla \log p_t(x)`.

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
        **kwargs: dict[str, Any],
    ) -> tuple[DataType, torch.Tensor]:
        """Compute the drift term of the environment's dynamics.

        Parameters
        ----------
        x : torch.Tensor, shape (n, *action_shape)
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to the base model (e.g. text embedding or class
            label).

        Returns
        -------
        drift : torch.Tensor, shape (n, *action_shape)
            The drift term at state x and time t.

        running_cost : torch.Tensor, shape (n,)
            Running cost :math:`L(x_t, t)` of the policy for the given (state, timestep)-pair.
        """
        alpha = self.scheduler.alpha(t)
        alpha_dot = self.scheduler.alpha_dot(t)
        sigma = self.scheduler.sigma(t)

        action = self.policy(x, t, **kwargs)

        control = x.zeros_like()
        if self.is_policy_set:
            action_base = self.base_model.forward(x, t, **kwargs)
            control = action - action_base

        if self.additive_policy is not None:
            action_add = self.additive_policy(x, t, **kwargs)
            action += action_add
            control += action_add

        a = alpha_dot / alpha
        b = sigma * sigma

        return a * x + b * action, 0.5 * (control * control).batch_sum()
