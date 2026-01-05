"""Pre-trained continuous FlowMol model, trained on GEOM-Drugs."""

from typing import Any

import dgl
import torch

try:
    import flowmol
    from flowmol.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask
except ImportError as exc:  # pragma: no cover - only hit when dependency is missing
    raise ImportError(
        "FlowMol is required for molecule environments. "
        "Install it manually, e.g. "
        "`pip install git+https://github.com/cristianpjensen/FlowMol.git@a666676c2f3835fc410dede22eb41c5c7c4f2eb8`."
    ) from exc

from flowgym import BaseModel, ConstantNoiseSchedule, CosineScheduler, Scheduler
from flowgym.molecules.types import FlowGraph
from flowgym.registry import base_model_registry
from flowgym.types import FlowTensor


class FlowMolBaseModel(BaseModel[FlowGraph]):
    """Pre-trained FlowMol on GEOM-Drugs and QM9."""

    output_type = "endpoint"

    def __init__(
        self,
        model_name: str,
        scheduler_params: tuple[float, float, float, float],
        device: torch.device,
    ):
        super().__init__(device)

        self.model = flowmol.load_pretrained(model_name).to(device)
        self._scheduler = FlowMolScheduler(scheduler_params)
        self._scheduler.noise_schedule = ConstantNoiseSchedule(0)

    @property
    def scheduler(self) -> "FlowMolScheduler":
        """Scheduler used for sampling."""
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowGraph, dict[str, Any]]:
        """Sample n datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : FlowGraph
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

        return FlowGraph(g, ue_mask, n_idx, e_idx), kwargs

    def postprocess(self, x: FlowGraph) -> FlowGraph:
        """Re-name features from x_t to x_1."""
        g = x.graph
        for key in list(g.ndata.keys()):
            if key.endswith("_t"):
                g.ndata[key[:-2] + "_1"] = g.ndata.pop(key)

        for key in list(g.edata.keys()):
            if key.endswith("_t"):
                g.edata[key[:-2] + "_1"] = g.edata.pop(key)

        # To enable usage with SampledMolecule from flowmol
        g.edata["ue_mask"] = x.ue_mask
        return FlowGraph(g, x.ue_mask, x.n_idx, x.e_idx)

    def forward(self, x: FlowGraph, t: torch.Tensor, **kwargs: Any) -> FlowGraph:
        r"""Compute the endpoint vector field :math:`\hat{x_1}(x, t)`."""
        output = self.model.vector_field(
            x.graph,
            t,
            x.n_idx,
            x.ue_mask,
            apply_softmax=True,
            remove_com=True,
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

        return FlowGraph(out_graph, x.ue_mask, x.n_idx, x.e_idx)


class FlowMolScheduler(Scheduler[FlowGraph]):
    """Scheduler for the GEOM-Drugs dataset."""

    def __init__(
        self,
        cosine_params: tuple[float, float, float, float] = (1.0, 2.0, 2.0, 2.0),
    ) -> None:
        self.schedulers = {
            "x_t": CosineScheduler(cosine_params[0]),
            "a_t": CosineScheduler(cosine_params[1]),
            "c_t": CosineScheduler(cosine_params[2]),
            "e_t": CosineScheduler(cosine_params[3]),
        }
        self.scheduler_order = ["x_t", "a_t", "c_t", "e_t"]

    def _alpha(self, x: FlowGraph, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(t.shape[0], len(self.schedulers), device=t.device, dtype=t.dtype)
        for idx, key in enumerate(self.scheduler_order):
            val = self.schedulers[key].alpha(
                FlowTensor(torch.zeros(x.graph.batch_size, 1, device=x.device)), t
            )
            out[:, idx] = val.data.squeeze(-1)

        return out

    def alpha(self, x: FlowGraph, t: torch.Tensor) -> FlowGraph:
        r""":math:`\alpha_t`."""
        g = x.graph
        res = x._get_empty_graph()

        alphas = self._alpha(x, t)
        for idx, key in enumerate(self.scheduler_order):
            if key in g.ndata:
                t_alpha = alphas[:, idx].unsqueeze(-1)
                res.ndata[key] = t_alpha[x.n_idx]
            elif key in g.edata:
                t_alpha = alphas[:, idx].unsqueeze(-1)
                res.edata[key] = t_alpha[x.e_idx]
            else:
                raise ValueError(f"Key {key} not found in graph data.")

        return FlowGraph(res, x.ue_mask, x.n_idx, x.e_idx)

    def _alpha_dot(self, x: FlowGraph, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(t.shape[0], len(self.schedulers), device=t.device, dtype=t.dtype)
        for idx, key in enumerate(self.scheduler_order):
            val = self.schedulers[key].alpha_dot(
                FlowTensor(torch.zeros(x.graph.batch_size, 1, device=x.device)), t
            )
            out[:, idx] = val.data.squeeze(-1)

        return out

    def alpha_dot(self, x: FlowGraph, t: torch.Tensor) -> FlowGraph:
        r""":math:`\dot{\alpha}_t`."""
        g = x.graph
        res = x._get_empty_graph()

        alpha_dots = self._alpha_dot(x, t)
        for idx, key in enumerate(self.scheduler_order):
            if key in g.ndata:
                t_alpha_dot = alpha_dots[:, idx].unsqueeze(-1)
                res.ndata[key] = t_alpha_dot[x.n_idx]
            elif key in g.edata:
                t_alpha_dot = alpha_dots[:, idx].unsqueeze(-1)
                res.edata[key] = t_alpha_dot[x.e_idx]
            else:
                raise ValueError(f"Key {key} not found in graph data.")

        return FlowGraph(res, x.ue_mask, x.n_idx, x.e_idx)


@base_model_registry.register("molecules/flowmol_geom")
class GEOMBaseModel(FlowMolBaseModel):
    """Pre-trained continuous flow matching model, trained on GEOM-Drugs."""

    def __init__(self, device: torch.device):
        super().__init__("geom_gaussian", (1.0, 2.0, 2.0, 2.0), device)


@base_model_registry.register("molecules/flowmol_qm9")
class QM9BaseModel(FlowMolBaseModel):
    """Pre-trained continuous flow matching model, trained on QM9."""

    def __init__(self, device: torch.device):
        super().__init__("qm9_gaussian", (1.0, 2.0, 2.0, 1.5), device)
