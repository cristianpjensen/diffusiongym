"""Types for molecular graphs in Flow Gym."""

from typing import Optional, Sequence, Union

import dgl
import torch
from typing_extensions import Self

from flowgym.types import BinaryOp, FlowMixin, UnaryOp


def construct_ue_mask(g: dgl.DGLGraph) -> torch.Tensor:
    """Construct a mask indicating upper edges in the graph."""
    ul_pattern = torch.tensor([1, 0], device=g.device).repeat(g.batch_size)
    n_edges_pattern = (g.batch_num_edges() / 2).int().repeat_interleave(2)
    return ul_pattern.repeat_interleave(n_edges_pattern).bool()


def construct_n_idx(g: dgl.DGLGraph) -> torch.Tensor:
    """Construct a tensor which maps each node to its graph index in the batch."""
    return torch.repeat_interleave(
        torch.arange(g.batch_size, device=g.device),
        g.batch_num_nodes(),
    )


def construct_e_idx(g: dgl.DGLGraph) -> torch.Tensor:
    """Construct a tensor which maps each edge to its graph index in the batch."""
    return torch.repeat_interleave(
        torch.arange(g.batch_size, device=g.device),
        g.batch_num_edges(),
    )


class FlowGraph(FlowMixin):
    """A wrapper around DGLGraph that supports required factory methods.

    Parameters
    ----------
    graph : dgl.DGLGraph
        The graph to wrap.

    ue_mask : Optional[torch.Tensor], optional
        Mask indicating upper edges in the graph, by default None

    n_idx : Optional[torch.Tensor], optional
        Tensor mapping each node to its graph index in the batch, by default None

    e_idx : Optional[torch.Tensor], optional
        Tensor mapping each edge to its graph index in the batch, by default None
    """

    def __init__(
        self,
        graph: dgl.DGLGraph,
        ue_mask: Optional[torch.Tensor] = None,
        n_idx: Optional[torch.Tensor] = None,
        e_idx: Optional[torch.Tensor] = None,
    ):
        if ue_mask is None:
            ue_mask = construct_ue_mask(graph)
        if n_idx is None:
            n_idx = construct_n_idx(graph)
        if e_idx is None:
            e_idx = construct_e_idx(graph)

        self.graph = graph
        self.ue_mask = ue_mask
        self.n_idx = n_idx
        self.e_idx = e_idx

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={self.graph.num_nodes()}, num_edges={self.graph.num_edges()}, batch_size={len(self)})"

    @property
    def device(self) -> torch.device:
        return self.graph.device

    def to(self, device: torch.device | str) -> Self:
        return self.__class__(
            self.graph.to(device),
            self.ue_mask.to(device),
            self.n_idx.to(device),
            self.e_idx.to(device),
        )

    def __len__(self) -> int:
        return int(self.graph.batch_size)

    def __getitem__(self, idx: Union[int, slice]) -> Self:
        if isinstance(idx, int):
            n = len(self)

            # Faster to use slice_batch if we only want one item
            if idx < 0:
                idx += n

            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for batch size {n}")

            return self.__class__(dgl.slice_batch(self.graph, idx))

        if isinstance(idx, slice):
            graphs = dgl.unbatch(self.graph)
            selected_graphs = graphs[idx]

            if not selected_graphs:
                raise ValueError("The slice resulted in an empty graph sequence.")

            return self.__class__(dgl.batch(selected_graphs))

        raise TypeError(f"Invalid index type: {type(idx)}")

    @classmethod
    def collate(cls, items: Sequence[Self]) -> Self:
        if not items:
            raise ValueError("Cannot collate an empty sequence")

        return cls(dgl.batch([item.graph for item in items]))

    def _get_empty_graph(self) -> dgl.DGLGraph:
        """Get an empty graph with the same structure as Self."""
        # Clone the graph structure
        empty_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

        # Preserve batch information
        if self.graph.batch_size > 1:
            empty_graph.set_batch_num_nodes(self.graph.batch_num_nodes())
            empty_graph.set_batch_num_edges(self.graph.batch_num_edges())

        return empty_graph

    def apply(self, op: UnaryOp) -> Self:
        res = self._get_empty_graph()

        for key, val in self.graph.ndata.items():
            if isinstance(val, torch.Tensor):
                res.ndata[key] = op(val)

        for key, val in self.graph.edata.items():
            if isinstance(val, torch.Tensor):
                res.edata[key] = op(val)

        return self.__class__(res, self.ue_mask, self.n_idx, self.e_idx)

    def combine(self, other: Union[Self, float, torch.Tensor], op: BinaryOp) -> Self:
        res = self._get_empty_graph()

        if isinstance(other, FlowGraph):
            for key, val in self.graph.ndata.items():
                if key in other.graph.ndata:
                    res.ndata[key] = op(val, other.graph.ndata[key])  # type: ignore
                else:
                    res.ndata[key] = val

            for key, val in self.graph.edata.items():
                if key in other.graph.edata:
                    res.edata[key] = op(val, other.graph.edata[key])  # type: ignore
                else:
                    res.edata[key] = val
        else:
            for key, val in self.graph.ndata.items():
                res.ndata[key] = op(val, other)  # type: ignore

            for key, val in self.graph.edata.items():
                res.edata[key] = op(val, other)  # type: ignore

        return self.__class__(res, self.ue_mask, self.n_idx, self.e_idx)

    def aggregate(self, reduction: str = "mean") -> torch.Tensor:
        batch_size = len(self)
        summed = torch.zeros(batch_size, device=self.graph.device)

        # Initialize counts if we need to calculate the mean later
        counts = None
        if reduction == "mean":
            counts = torch.zeros(batch_size, device=self.graph.device)

        for _, val in self.graph.ndata.items():
            if isinstance(val, torch.Tensor):
                aggregated = torch.zeros(batch_size, *val.shape[1:], device=val.device, dtype=val.dtype)
                aggregated.index_add_(0, self.n_idx, val)
                summed += aggregated.sum(dim=-1)

                # Track number of elements added
                if counts is not None:
                    num_elements = val[0].numel()
                    item_counts = torch.zeros(batch_size, device=val.device)
                    ones = torch.ones(val.size(0), device=val.device)
                    item_counts.index_add_(0, self.n_idx, ones)
                    counts += item_counts * num_elements

        for _, val in self.graph.edata.items():
            if isinstance(val, torch.Tensor):
                aggregated = torch.zeros(batch_size, *val.shape[1:], device=val.device, dtype=val.dtype)
                aggregated.index_add_(0, self.e_idx, val)
                summed += aggregated.sum(dim=-1)

                # Track number of elements added
                if counts is not None:
                    num_elements = val[0].numel()
                    item_counts = torch.zeros(batch_size, device=val.device)
                    ones = torch.ones(val.size(0), device=val.device)
                    item_counts.index_add_(0, self.e_idx, ones)
                    counts += item_counts * num_elements

        if counts is not None:
            # Avoid division by zero for empty graphs
            return summed / counts.clamp(min=1)

        return summed

    def randn_like(self) -> Self:
        out = super().randn_like()

        # Remove COM
        init_coms = dgl.readout_nodes(out.graph, feat="x_t", op="mean")
        out.graph.ndata["x_t"] = out.graph.ndata["x_t"] - init_coms[out.n_idx]

        # Also make sure that both sides of edges are equivalent
        out.graph.edata["e_t"][~out.ue_mask] = out.graph.edata["e_t"][out.ue_mask]

        return out
