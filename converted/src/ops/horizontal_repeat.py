import torch
import torch.nn as nn
from torch.func import vmap, functional_call, stack_module_state

from ops.subflow import Subflow


class HorizontalRepeat(nn.Module):
    """Runs N parallel copies of subgraph on same input via vmap.

    Creates N independent Subflow instances with fresh modules.
    Forward uses vmap + functional_call for batched parallel execution.

    Join is hardcoded to **concat on dim=-1**: head outputs are stacked,
    moved to last position, and flattened. Output shape: [batch, ..., n * d_head].

    To use a different join (add, mean, max, etc.), modify forward() or create
    a new op — HorizontalRepeat does not expose join_type as a parameter.
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
        # Reference module on meta device for functional_call.
        # Stored as plain attribute (not registered submodule) so Lightning/.to()
        # won't try to move meta tensors — base is structure-only, no real weights.
        _base = Subflow(entry_node=entry_node, internal_nodes=internal_nodes)
        _base.to("meta")
        object.__setattr__(self, "base", _base)

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
