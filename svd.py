from scipy.sparse.linalg import svds
import numpy as np
import torch


J = torch.load(f"jacobians/layer_9.pt")
J_np = J.numpy()
U, S, Vh = svds(J_np, k=5)
print("Top-k singular values:")
for i, s in enumerate(S):
    print(f"{i+1}: {s:.4f}")
del U, S, Vh, J_np, J


