"""Types for molecular graphs in Flow Gym."""

from typing import Any, Callable, Union

import dgl
import torch
from torch_scatter import scatter


class FGGraph:
    """A wrapper around DGLGraph that supports required factory methods.

    Parameters
    ----------
    graph : dgl.DGLGraph
        The graph to wrap.

    ue_mask : torch.Tensor, shape (num_edges,)
        Mask indicating upper edges in the graph.

    n_idx : torch.Tensor, shape (num_nodes,)
        Batch indices for nodes.

    e_idx : torch.Tensor, shape (num_edges,)
        Batch indices for edges.
    """

    def __init__(
        self,
        graph: dgl.DGLGraph,
        ue_mask: torch.Tensor,
        n_idx: torch.Tensor,
        e_idx: torch.Tensor,
    ):
        self.graph = graph
        self.ue_mask = ue_mask
        self.n_idx = n_idx
        self.e_idx = e_idx

    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "graph":
            object.__setattr__(self, name, value)
        else:
            setattr(self.graph, name, value)

    def _get_empty_graph(self) -> dgl.DGLGraph:
        """Get an empty graph with the same structure as Self."""
        # Clone the graph structure
        empty_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

        # Preserve batch information
        if self.graph.batch_size > 1:
            empty_graph.set_batch_num_nodes(self.graph.batch_num_nodes())
            empty_graph.set_batch_num_edges(self.graph.batch_num_edges())

        return empty_graph

    def _apply_unary_op(self, op: Callable[[torch.Tensor], torch.Tensor]) -> "FGGraph":
        """Apply a unary operation to graph data.

        Parameters
        ----------
        op : (torch.Tensor) -> torch.Tensor
            Unary operation to apply.

        Returns
        -------
        output : FGGraph
            New graph with operation applied.
        """
        res = self._get_empty_graph()

        for key, val in self.graph.ndata.items():
            if isinstance(val, torch.Tensor):
                res.ndata[key] = op(val)

        for key, val in self.graph.edata.items():
            if isinstance(val, torch.Tensor):
                res.edata[key] = op(val)

        return FGGraph(res, self.ue_mask, self.n_idx, self.e_idx)

    def _apply_binary_op(
        self,
        other: Union["FGGraph", float, torch.Tensor],
        op: Callable[[Any, Any], Any],
    ) -> "FGGraph":
        """Apply a binary operation to graph data.

        Parameters
        ----------
        other : FGGraph, float, or torch.Tensor
            The other operand.

        op : (any, any) -> any
            Binary operation to apply.

        Returns
        -------
        output : FGGraph
            New graph with operation applied.
        """
        res = self._get_empty_graph()

        if isinstance(other, FGGraph):
            for key, val in self.graph.ndata.items():
                if key in other.graph.ndata:
                    res.ndata[key] = op(val, other.graph.ndata[key])
                else:
                    res.ndata[key] = val

            for key, val in self.graph.edata.items():
                if key in other.graph.edata:
                    res.edata[key] = op(val, other.graph.edata[key])
                else:
                    res.edata[key] = val
        else:
            for key, val in self.graph.ndata.items():
                res.ndata[key] = op(val, other)

            for key, val in self.graph.edata.items():
                res.edata[key] = op(val, other)

        return FGGraph(res, self.ue_mask, self.n_idx, self.e_idx)

    def __add__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        return self._apply_binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        return self._apply_binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        return self._apply_binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        return self._apply_binary_op(other, lambda a, b: a / b)

    def __radd__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        return self._apply_binary_op(other, lambda a, b: b + a)

    def __rsub__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        return self._apply_binary_op(other, lambda a, b: b - a)

    def __rmul__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        return self._apply_binary_op(other, lambda a, b: b * a)

    def __pow__(self, power: float) -> "FGGraph":
        return self._apply_binary_op(power, lambda a, b: a**b)

    def __neg__(self) -> "FGGraph":
        return self._apply_unary_op(lambda x: -x)

    def randn_like(self) -> "FGGraph":
        """Generate random normal noise with the same shape and type as self."""
        return self._apply_unary_op(torch.randn_like)

    def ones_like(self) -> "FGGraph":
        """Generate ones with the same shape and type as self."""
        return self._apply_unary_op(torch.ones_like)

    def zeros_like(self) -> "FGGraph":
        """Generate zeros with the same shape and type as self."""
        return self._apply_unary_op(torch.zeros_like)

    def batch_sum(self) -> torch.Tensor:
        """Sum over all dimensions except the first (batch) dimension.

        Returns
        -------
        sum : torch.Tensor, shape (batch_size,)
        """
        summed = torch.zeros(self.graph.batch_size, device=self.graph.device)
        for _, val in self.graph.ndata.items():
            if isinstance(val, torch.Tensor):
                summed += scatter(val, self.n_idx, dim=0, reduce="sum").sum(dim=-1)

        for _, val in self.graph.edata.items():
            if isinstance(val, torch.Tensor):
                summed += scatter(val, self.e_idx, dim=0, reduce="sum").sum(dim=-1)

        return summed

    def to_device(self, device: torch.device | str) -> "FGGraph":
        """Move the graph to the specified device.

        Returns
        -------
        output : FGGraph
            A copy of self on the specified device.
        """
        return FGGraph(
            self.graph.to(device),
            self.ue_mask.to(device),
            self.n_idx.to(device),
            self.e_idx.to(device),
        )

    def with_requires_grad(self) -> "FGGraph":
        """Enable gradient tracking for all tensors in the graph.

        Returns
        -------
        output : FGGraph
            New graph with requires_grad=True for all tensors.
        """
        res = self._get_empty_graph()

        for key, val in self.graph.ndata.items():
            if isinstance(val, torch.Tensor):
                res.ndata[key] = val.requires_grad_(True)
            else:
                res.ndata[key] = val

        for key, val in self.graph.edata.items():
            if isinstance(val, torch.Tensor):
                res.edata[key] = val.requires_grad_(True)
            else:
                res.edata[key] = val

        return FGGraph(res, self.ue_mask, self.n_idx, self.e_idx)

    def gradient(self, x: torch.Tensor) -> "FGGraph":
        """Compute gradients of x with respect to all tensors in the graph.

        Parameters
        ----------
        x : torch.Tensor
            The tensor to compute gradients from (typically a loss or output).

        Returns
        -------
        output : FGGraph
            New graph containing gradients of x with respect to each tensor.
        """
        res = self._get_empty_graph()

        for n_feat, val in self.graph.ndata.items():
            if isinstance(val, torch.Tensor):
                grad = torch.autograd.grad(
                    x, val, grad_outputs=torch.ones_like(x), retain_graph=True
                )[0]
                if grad is None:
                    grad = torch.zeros_like(val)

                res.ndata[n_feat] = grad

        for e_feat, val in self.graph.edata.items():
            if isinstance(val, torch.Tensor):
                grad = torch.autograd.grad(
                    x, val, grad_outputs=torch.ones_like(x), retain_graph=True
                )[0]
                if grad is None:
                    grad = torch.zeros_like(val)

                res.edata[e_feat] = grad

        return FGGraph(res, self.ue_mask, self.n_idx, self.e_idx)
