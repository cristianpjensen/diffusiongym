"""Types for molecular graphs in Flow Gym."""

from typing import Any, Callable, Union

import dgl
import torch


class FGGraph:
    """A wrapper around DGLGraph that supports required factory methods."""

    def __init__(self, graph: dgl.DGLGraph):
        """Create a new FGGraph from a DGLGraph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The graph to wrap.
        """
        object.__setattr__(self, "graph", graph)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped graph."""
        return getattr(self.graph, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to the wrapped graph."""
        if name == "graph":
            object.__setattr__(self, name, value)
        else:
            setattr(self.graph, name, value)

    def _apply_binary_op(
        self, other: Union["FGGraph", float, torch.Tensor], op: Callable[[Any, Any], Any]
    ) -> "FGGraph":
        """Apply a binary operation to graph data.

        Parameters
        ----------
        other : FGGraph, float, or torch.Tensor
            The other operand.
        op : callable
            Binary operation to apply (e.g., operator.add, operator.sub).

        Returns
        -------
        FGGraph
            New graph with operation applied.
        """
        if isinstance(other, FGGraph):
            # Create a copy of self.graph
            result_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

            # Apply operation to all ndata from self
            for key in self.graph.ndata.keys():
                if key in other.graph.ndata:
                    result_graph.ndata[key] = op(self.graph.ndata[key], other.graph.ndata[key])
                else:
                    result_graph.ndata[key] = self.graph.ndata[key]

            # Apply operation to all edata from self
            for key in self.graph.edata.keys():
                if key in other.graph.edata:
                    result_graph.edata[key] = op(self.graph.edata[key], other.graph.edata[key])
                else:
                    result_graph.edata[key] = self.graph.edata[key]

            return FGGraph(result_graph)

        # Apply operation with scalar/tensor to all data
        result_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

        for key in self.graph.ndata.keys():
            result_graph.ndata[key] = op(self.graph.ndata[key], other)

        for key in self.graph.edata.keys():
            result_graph.edata[key] = op(self.graph.edata[key], other)

        return FGGraph(result_graph)

    def __add__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        """Add graph data together."""
        return self._apply_binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        """Subtract graph data."""
        return self._apply_binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        """Multiply graph data."""
        return self._apply_binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        """Divide graph data."""
        return self._apply_binary_op(other, lambda a, b: a / b)

    def __neg__(self) -> "FGGraph":
        """Negate graph data."""
        result_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

        for key in self.graph.ndata.keys():
            result_graph.ndata[key] = -self.graph.ndata[key]

        for key in self.graph.edata.keys():
            result_graph.edata[key] = -self.graph.edata[key]

        return FGGraph(result_graph)

    def __radd__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        """Right addition."""
        return self._apply_binary_op(other, lambda a, b: b + a)

    def __rsub__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        """Right subtraction."""
        return self._apply_binary_op(other, lambda a, b: b - a)

    def __rmul__(self, other: Union["FGGraph", float, torch.Tensor]) -> "FGGraph":
        """Right multiplication."""
        return self._apply_binary_op(other, lambda a, b: b * a)

    def __pow__(self, power: float) -> "FGGraph":
        """Raise graph data to a power."""
        return self._apply_binary_op(power, lambda a, b: a**b)

    def randn_like(self) -> "FGGraph":
        """Generate random normal noise with the same shape and type as self."""
        result_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

        for key in self.graph.ndata.keys():
            result_graph.ndata[key] = torch.randn_like(self.graph.ndata[key])

        for key in self.graph.edata.keys():
            result_graph.edata[key] = torch.randn_like(self.graph.edata[key])

        return FGGraph(result_graph)

    def ones_like(self) -> "FGGraph":
        """Generate ones with the same shape and type as self."""
        result_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

        for key in self.graph.ndata.keys():
            result_graph.ndata[key] = torch.ones_like(self.graph.ndata[key])

        for key in self.graph.edata.keys():
            result_graph.edata[key] = torch.ones_like(self.graph.edata[key])

        return FGGraph(result_graph)

    def zeros_like(self) -> "FGGraph":
        """Generate zeros with the same shape and type as self."""
        result_graph = dgl.graph(self.graph.edges(), num_nodes=self.graph.num_nodes())

        for key in self.graph.ndata.keys():
            result_graph.ndata[key] = torch.zeros_like(self.graph.ndata[key])

        for key in self.graph.edata.keys():
            result_graph.edata[key] = torch.zeros_like(self.graph.edata[key])

        return FGGraph(result_graph)

    def batch_sum(self) -> torch.Tensor:
        """Sum over all dimensions except the first (batch) dimension.

        Returns
        -------
        sum : torch.Tensor, shape (batch_size,)
        """
        raise NotImplementedError
