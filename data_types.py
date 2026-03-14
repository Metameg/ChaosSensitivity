import torch
from dataclasses import dataclass

@dataclass
class LayerInputs:
    hidden_states: torch.Tensor
    causal_mask: torch.Tensor
    position_ids: torch.Tensor
    position_embeddings: tuple
    cache_position: torch.Tensor

@dataclass
class LyapunovResult:
    exponents: list[float]
    per_layer_log_stretches: list[float]
    lyapunov_vector: torch.Tensor
    layer_indices: list[int]