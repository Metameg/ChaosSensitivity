from scipy.sparse.linalg import svds
import numpy as np
import torch


J = torch.load(f"full_jacobians/layer_1/shard_0.pt")
J_np = J.numpy()
U, S, Vh = svds(J_np, k=1)
print("Top-k singular values:")
for i, s in enumerate(S):
    print(f"{i+1}: {s:.4f}")
del U, S, Vh, J_np, J


