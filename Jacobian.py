import dataclasses
import torch
from tqdm import tqdm
from pathlib import Path


class Jacobian:
    """
    Computes and saves the Jacobian of each hidden layer in chunks.

    Usage:
        jac = Jacobian(mmodel, layer_inputs, save_dir="jacobians")
        jac.compute()
    """

    def __init__(
        self,
        mmodel,
        layer_inputs,
        save_dir: str = "jacobians",
        chunk_size: int = 256,
        power_iter_steps: int = 20
    ):
        self.mmodel = mmodel
        self.layer_inputs = layer_inputs
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size
        self.power_iter_steps = power_iter_steps

        self.layers = mmodel.model.model.layers
        self.seq_len = layer_inputs.hidden_states.shape[1]
        self.J_size = self.seq_len * mmodel.model.config.hidden_size

        # Tracks {layer_idx: [shard_paths]} after compute() is called
        self.shard_manifest: dict[int, list[Path]] = {}
        self.spectral_norms: dict[int, float] = {}

    def _compute_chunk(self, layer_idx: int, output_start: int, output_end: int) -> torch.Tensor:
        """Compute a single Jacobian chunk for the given output range."""
        h_in = self.layer_inputs.hidden_states.detach().requires_grad_(True)
        inputs_for_jacobian = dataclasses.replace(self.layer_inputs, hidden_states=h_in)

        def forward_partial(h):
            updated_inputs = dataclasses.replace(inputs_for_jacobian, hidden_states=h)
            return (
                self.mmodel.forward_layer(layer_idx, updated_inputs, no_grad=False)
                .hidden_states[0]
                .reshape(-1)[output_start:output_end]
            )

        J = torch.autograd.functional.jacobian(forward_partial, h_in)
        return J.reshape(output_end - output_start, self.J_size)
    
    
    def _power_iteration_from_disk(self, layer_idx: int, steps: int | None = None) -> float:
        """
        Estimate sv₁ via power iteration, loading one shard at a time.

        Each iteration:
          Forward pass:  u = J @ v  (accumulate chunk contributions row-wise)
          Backward pass: v = J^T @ u (accumulate chunk contributions, sum cols)
        """
        steps = steps or self.power_iter_steps
        paths = self.shard_manifest[layer_idx]

        # Random unit vector in input space
        v = torch.randn(self.J_size)
        v = v / v.norm()

        for _ in range(steps):
            # Forward: u = J @ v, built chunk by chunk
            u_chunks = []
            for path in paths:
                J_chunk = torch.load(path, weights_only=True)   # [chunk_rows, J_size_in]
                u_chunks.append(J_chunk @ v)
                del J_chunk
            u = torch.cat(u_chunks, dim=0)                      # [J_size_out]
            u = u / u.norm()

            # Backward: v = Jᵀ @ u, accumulated across chunks
            v = torch.zeros(self.J_size)
            offset = 0
            for path in paths:
                J_chunk = torch.load(path, weights_only=True)   # [chunk_rows, J_size_in]
                n_rows = J_chunk.shape[0]
                v += J_chunk.T @ u[offset: offset + n_rows]
                offset += n_rows
                del J_chunk
            v = v / v.norm()

        # Final σ₁ estimate: ||J @ v||
        u_chunks = []
        for path in paths:
            J_chunk = torch.load(path, weights_only=True)
            u_chunks.append(J_chunk @ v)
            del J_chunk
        sigma_1 = torch.cat(u_chunks, dim=0).norm().item()

        return sigma_1
    

    def _reload_manifest_from_disk(self, layer_idx: int) -> None:
        """Repopulate shard_manifest for a layer by scanning the save directory."""
        layer_dir = self.save_dir / f"layer_{layer_idx}"
        if not layer_dir.exists():
            raise FileNotFoundError(f"No shard directory found at {layer_dir}")
        paths = sorted(layer_dir.glob("shard_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not paths:
            raise FileNotFoundError(f"No shards found in {layer_dir}")
        self.shard_manifest[layer_idx] = paths


    def compute(self) -> None:
        """Compute Jacobians for all layers, saving each chunk to disk."""
        for l, _ in enumerate(tqdm(self.layers, desc="Layers")):
            layer_dir = self.save_dir / f"layer_{l}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            self.shard_manifest[l] = []

            chunk_indices = range(0, self.J_size, self.chunk_size)
            for chunk_idx, output_start in enumerate(tqdm(chunk_indices, desc=f"  Layer {l} chunks", leave=False)):
                output_end = min(output_start + self.chunk_size, self.J_size)
                J = self._compute_chunk(l, output_start, output_end)
                shard_path = self.save(J, layer_idx=l, chunk_idx=chunk_idx)
                self.shard_manifest[l].append(shard_path)

                del J
                torch.cuda.empty_cache()

    def save(self, J: torch.Tensor, layer_idx: int, chunk_idx: int) -> Path:
        """Save a single Jacobian shard to disk. Returns the path written."""
        path = self.save_dir / f"layer_{layer_idx}" / f"shard_{chunk_idx}.pt"
        torch.save(J.float().cpu(), path)
        return path

    def load(self, layer_idx: int) -> torch.Tensor:
        """Reconstruct the full Jacobian for a layer from its saved shards."""
        paths = self.shard_manifest.get(layer_idx)
        if not paths:
            raise ValueError(f"No shards found for layer {layer_idx}. Run compute() first.")
        shards = [torch.load(p) for p in paths]
        return torch.cat(shards, dim=0)
    
    def spectral_norm_from_disk(self, layer_idx: int, power_iter_steps: int | None = None) -> float:
        """
        Estimate σ₁ for a layer from previously saved shards.
        Useful for re-running without recomputing the Jacobian.
        """
        self._reload_manifest_from_disk(layer_idx)
        steps = power_iter_steps or self.power_iter_steps
        return self._power_iteration_from_disk(layer_idx, steps)


    