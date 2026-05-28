import torch.nn as nn

from ops.subflow import Subflow


class Repeat(nn.Module):
    """Repeats subgraph N times with independent module instances.

    Creates N independent Subflow instances chained in nn.Sequential.
    Each instance gets fresh modules (Hydra instantiate creates new objects).
    """

    def __init__(self, entry_node: str, internal_nodes: dict, iterations: int = 1, **kwargs):
        super().__init__()
        self.net = nn.Sequential(*[
            Subflow(entry_node=entry_node, internal_nodes=internal_nodes)
            for _ in range(iterations)
        ])

    def forward(self, x):
        return self.net(x)
