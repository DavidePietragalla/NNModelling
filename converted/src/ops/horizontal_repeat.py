import torch
import torch.nn as nn
from torch.func import vmap, functional_call, stack_module_state

from ops.subflow import Subflow


class HorizontalRepeat(nn.Module):
    """Runs N parallel copies of subgraph on same input via vmap.

    Creates N independent Subflow instances with fresh modules.
    Forward uses vmap + functional_call for batched parallel execution.

    Output shape: [batch, ..., n * d_out]
    The stacked head dimension is moved to last position and flattened,
    effectively concatenating all head outputs along the feature dimension.
    """

    def __init__(self, entry_node: str, internal_nodes: dict, n: int = 1, **kwargs):
        super().__init__()
        if n < 1:
            raise ValueError(f"HorizontalRepeat n must be >= 1, got {n}")

        self.n = n
        self.heads = nn.ModuleList([
            Subflow(entry_node=entry_node, internal_nodes=internal_nodes)
            for _ in range(n)
        ])
        # Reference module on meta device for functional_call
        self.base = Subflow(entry_node=entry_node, internal_nodes=internal_nodes)
        self.base.to("meta")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n == 1:
            return self.heads[0](x)

        params, buffers = stack_module_state(list(self.heads))

        def forward_single(p, b, x):
            return functional_call(self.base, (p, b), x)

        out = vmap(forward_single, in_dims=(0, 0, None))(params, buffers, x)
        # out shape: [n, batch, ..., d_head]
        # Reshape to: [batch, ..., n * d_head]
        out = out.moveaxis(0, -2)
        out = out.reshape(*out.shape[:-2], -1)
        return out
