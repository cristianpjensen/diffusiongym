"""Pre-trained continuous FlowMol model, trained on GEOM-Drugs."""

from typing import Any

import dgl
import flowmol
import torch
from flowmol.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask

from flow_gym import BaseModel, ConstantNoiseSchedule, CosineScheduler, Scheduler
from flow_gym.molecules.types import FGGraph
from flow_gym.registry import base_model_registry


@base_model_registry.register("molecules/flowmol")
class FlowMolBaseModel(BaseModel[FGGraph]):
    """Pre-trained continuous flow matching model on GEOM-Drugs."""

    output_type = "endpoint"

    def __init__(self, device: torch.device):
        super().__init__(device)

        self.model = flowmol.load_pretrained("geom_gaussian").to(device)
        self._scheduler = GEOMScheduler()
        self._scheduler.noise_schedule = ConstantNoiseSchedule(0)

    @property
    def scheduler(self) -> "GEOMScheduler":
        """Scheduler used for sampling."""
        return self._scheduler

    def sample_p0(self, n: int) -> FGGraph:
        """Sample n datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : FGGraph
            Samples from the base distribution :math:`p_0`.

        Notes
        -----
        The base distribution :math:`p_0` is a standard Gaussian distribution.
        """
        n_atoms = self.model.sample_n_atoms(n)

        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=self.device)
            g.append(g_i)

        g = dgl.batch(g)
        ue_mask = get_upper_edge_mask(g)
        n_idx, e_idx = get_batch_idxs(g)
        g = self.model.sample_prior(g, n_idx, ue_mask)

        # Set all x_0 to x_t (this is what FlowMol expects)
        for key in list(g.ndata.keys()):
            if key.endswith("_0"):
                g.ndata[key[:-2] + "_t"] = g.ndata.pop(key)

        for key in list(g.edata.keys()):
            if key.endswith("_0"):
                g.edata[key[:-2] + "_t"] = g.edata.pop(key)

        return FGGraph(g, ue_mask, n_idx, e_idx)

    def postprocess(self, x: FGGraph) -> FGGraph:
        """Re-name features from x_t to x_1."""
        # TODO: Maybe some rdkit geometry relaxation

        g = x.graph
        for key in list(g.ndata.keys()):
            if key.endswith("_t"):
                g.ndata[key[:-2] + "_1"] = g.ndata.pop(key)

        for key in list(g.edata.keys()):
            if key.endswith("_t"):
                g.edata[key[:-2] + "_1"] = g.edata.pop(key)

        # To enable usage with SampledMolecule from flowmol
        g.edata["ue_mask"] = x.ue_mask
        return FGGraph(g, x.ue_mask, x.n_idx, x.e_idx)

    def forward(self, x: FGGraph, t: torch.Tensor, **kwargs: Any) -> FGGraph:
        r"""Compute the endpoint vector field :math:`\hat{x_1}(x, t)`."""
        output = self.model.vector_field(
            x.graph, t, x.n_idx, x.ue_mask, apply_softmax=True, remove_com=True, **kwargs
        )

        out_graph = x._get_empty_graph()
        for key in x.graph.ndata:
            out_graph.ndata[key] = output[key[:-2]]

        for key in x.graph.edata:
            data = x.graph.edata[key]
            if isinstance(data, torch.Tensor):
                # Output only contains upper edge data, need to expand to both upper and lower
                edge_data = torch.zeros_like(data)
                edge_data[x.ue_mask] = output[key[:-2]]  # Assign to upper edges
                edge_data[~x.ue_mask] = output[key[:-2]]  # Assign same values to lower edges
                out_graph.edata[key] = edge_data

        return FGGraph(out_graph, x.ue_mask, x.n_idx, x.e_idx)


class GEOMScheduler(Scheduler[FGGraph]):
    """Scheduler for the GEOM-Drugs dataset."""

    def __init__(self) -> None:
        self.schedulers = {
            "x_t": CosineScheduler(1.0),
            "a_t": CosineScheduler(2.0),
            "c_t": CosineScheduler(2.0),
            "e_t": CosineScheduler(2.0),
        }
        self.scheduler_order = ["x_t", "a_t", "c_t", "e_t"]

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(t.shape[0], len(self.schedulers), device=t.device, dtype=t.dtype)
        for idx, key in enumerate(self.scheduler_order):
            out[:, idx] = self.schedulers[key]._alpha(t).squeeze(-1)

        return out

    def alpha(self, x: FGGraph, t: torch.Tensor) -> FGGraph:
        r""":math:`\alpha_t`."""
        g = x.graph
        res = x._get_empty_graph()

        alphas = self._alpha(t)

        for idx, key in enumerate(self.scheduler_order):
            if key in g.ndata:
                t_alpha = alphas[:, idx].unsqueeze(-1)
                res.ndata[key] = t_alpha[x.n_idx]
            elif key in g.edata:
                t_alpha = alphas[:, idx].unsqueeze(-1)
                res.edata[key] = t_alpha[x.e_idx]
            else:
                raise ValueError(f"Key {key} not found in graph data.")

        return FGGraph(res, x.ue_mask, x.n_idx, x.e_idx)

    def _alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(t.shape[0], len(self.schedulers), device=t.device, dtype=t.dtype)
        for idx, key in enumerate(self.scheduler_order):
            out[:, idx] = self.schedulers[key]._alpha_dot(t).squeeze(-1)

        return out

    def alpha_dot(self, x: FGGraph, t: torch.Tensor) -> FGGraph:
        r""":math:`\dot{\alpha}_t`."""
        g = x.graph
        res = x._get_empty_graph()

        alpha_dots = self._alpha_dot(t)

        for idx, key in enumerate(self.scheduler_order):
            if key in g.ndata:
                t_alpha_dot = alpha_dots[:, idx].unsqueeze(-1)
                res.ndata[key] = t_alpha_dot[x.n_idx]
            elif key in g.edata:
                t_alpha_dot = alpha_dots[:, idx].unsqueeze(-1)
                res.edata[key] = t_alpha_dot[x.e_idx]
            else:
                raise ValueError(f"Key {key} not found in graph data.")

        return FGGraph(res, x.ue_mask, x.n_idx, x.e_idx)
