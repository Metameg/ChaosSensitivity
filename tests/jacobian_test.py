import pytest
import sys
import torch
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))
from Jacobian import Jacobian
from data_types import LayerInputs


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def small_dims():
    return {"seq_len": 2, "hidden_size": 3}


@pytest.fixture
def layer_inputs(small_dims):
    seq_len = small_dims["seq_len"]
    hidden_size = small_dims["hidden_size"]
    return LayerInputs(
        hidden_states=torch.randn(1, seq_len, hidden_size),
        causal_mask=None,
        position_ids=torch.arange(seq_len).unsqueeze(0),
        position_embeddings=(torch.randn(1, seq_len, hidden_size), torch.randn(1, seq_len, hidden_size)),
        cache_position=torch.arange(seq_len),
    )


@pytest.fixture
def mock_model(small_dims, layer_inputs):
    """
    A minimal mock of MyModel whose forward_layer applies a known
    linear transformation so we can compute the ground truth Jacobian
    analytically.
    """
    seq_len = small_dims["seq_len"]
    hidden_size = small_dims["hidden_size"]
    J_size = seq_len * hidden_size

    # Fixed weight matrix so the Jacobian is deterministic
    W = torch.randn(J_size, J_size)

    mmodel = MagicMock()
    mmodel.model.config.hidden_size = hidden_size
    mmodel.model.model.layers = [MagicMock()]  # single layer

    def forward_layer(layer_idx, inputs, no_grad=True):
        h = inputs.hidden_states.reshape(-1)
        out = (W @ h).reshape(1, seq_len, hidden_size)
        result = MagicMock()
        result.hidden_states = out
        return result

    mmodel.forward_layer.side_effect = forward_layer
    mmodel._W = W  # expose for ground truth checks
    return mmodel


@pytest.fixture
def jac(tmp_path, mock_model, layer_inputs):
    return Jacobian(
        mmodel=mock_model,
        layer_inputs=layer_inputs,
        save_dir=str(tmp_path / "jacobians"),
        chunk_size=2,
    )


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestChunking:
    def test_shards_are_saved_to_disk(self, jac):
        jac.compute()
        layer_dir = Path(jac.save_dir) / "layer_0"
        shards = list(layer_dir.glob("shard_*.pt"))
        assert len(shards) > 0

    def test_shard_manifest_is_populated(self, jac):
        jac.compute()
        assert 0 in jac.shard_manifest
        assert len(jac.shard_manifest[0]) > 0

    def test_shard_row_counts_sum_to_J_size(self, jac):
        jac.compute()
        total_rows = sum(
            torch.load(p, weights_only=True).shape[0]
            for p in jac.shard_manifest[0]
        )
        assert total_rows == jac.J_size

    def test_chunk_size_determines_number_of_shards(self, tmp_path, mock_model, layer_inputs):
        J_size = layer_inputs.hidden_states.shape[1] * mock_model.model.config.hidden_size
        for chunk_size in [1, 2, 3]:
            jac = Jacobian(mock_model, layer_inputs, save_dir=str(tmp_path / f"jac_{chunk_size}"), chunk_size=chunk_size)
            jac.compute()
            expected_shards = (J_size + chunk_size - 1) // chunk_size
            assert len(jac.shard_manifest[0]) == expected_shards


class TestChunkedVsAutograd:
    def test_chunked_jacobian_matches_autograd(self, jac):
        """Core correctness test: chunked shards should reconstruct the autograd Jacobian."""
        jac.compute()
        J_autograd = jac.compute_autograd(layer_idx=0)
        J_chunked = torch.cat(
            [torch.load(p, weights_only=True) for p in jac.shard_manifest[0]], dim=0
        )
        diff = (J_autograd - J_chunked).abs()
        assert diff.max().item() < 1e-5, f"Max diff too large: {diff.max().item():.2e}"
        assert diff.mean().item() < 1e-6, f"Mean diff too large: {diff.mean().item():.2e}"

    def test_full_chunk_matches_autograd(self, tmp_path, mock_model, layer_inputs):
        """When chunk_size == J_size, single shard should equal autograd Jacobian exactly."""
        J_size = layer_inputs.hidden_states.shape[1] * mock_model.model.config.hidden_size
        jac = Jacobian(mock_model, layer_inputs, save_dir=str(tmp_path / "full"), chunk_size=J_size)
        jac.compute()
        J_autograd = jac.compute_autograd(layer_idx=0)
        J_chunked = torch.load(jac.shard_manifest[0][0], weights_only=True)
        diff = (J_autograd - J_chunked).abs()
        assert diff.max().item() < 1e-5


class TestSpectralNorm:
    def test_spectral_norm_matches_svds(self, jac):
        """Power iteration σ₁ should match scipy svds to within tolerance."""
        from scipy.sparse.linalg import svds
        jac.compute()
        J_autograd = jac.compute_autograd(layer_idx=0)
        _, S, _ = svds(J_autograd.numpy(), k=1)
        sigma_svds = S[0]
        sigma_power = jac.spectral_norms[0]
        assert abs(sigma_power - sigma_svds) / (sigma_svds + 1e-8) < 1e-3, (
            f"Power iteration σ₁={sigma_power:.6f} vs svds σ₁={sigma_svds:.6f}"
        )

    def test_spectral_norm_from_disk_matches_compute(self, jac):
        """spectral_norm_from_disk() should reproduce the same result as compute()."""
        jac.compute()
        sigma_compute = jac.spectral_norms[0]
        sigma_disk = jac.spectral_norm_from_disk(layer_idx=0)
        assert abs(sigma_compute - sigma_disk) / (sigma_compute + 1e-8) < 1e-3

    def test_spectral_norm_known_matrix(self, tmp_path, layer_inputs):
        """
        For a diagonal weight matrix with known singular values,
        σ₁ should equal the largest diagonal entry.
        """
        seq_len = layer_inputs.hidden_states.shape[1]
        hidden_size = 3
        J_size = seq_len * hidden_size

        # Diagonal W with known singular values [5, 3, 2, 1, 1, 1]
        diag_values = torch.tensor([5.0, 3.0, 2.0, 1.0, 1.0, 1.0])
        W = torch.diag(diag_values)

        mmodel = MagicMock()
        mmodel.model.config.hidden_size = hidden_size
        mmodel.model.model.layers = [MagicMock()]

        def forward_layer(layer_idx, inputs, no_grad=True):
            h = inputs.hidden_states.reshape(-1)
            out = (W @ h).reshape(1, seq_len, hidden_size)
            result = MagicMock()
            result.hidden_states = out
            return result

        mmodel.forward_layer.side_effect = forward_layer

        jac = Jacobian(mmodel, layer_inputs, save_dir=str(tmp_path / "diag"), chunk_size=2)
        jac.compute()
        

        assert abs(jac.spectral_norms[0] - 5.0) < 0.1, (
            f"Expected σ₁ ≈ 5.0, got {jac.spectral_norms[0]:.4f}"
        )


class TestManifest:
    def test_reload_manifest_from_disk(self, jac):
        """_reload_manifest_from_disk should repopulate shard_manifest from saved files."""
        jac.compute()
        jac.shard_manifest.clear()
        jac._reload_manifest_from_disk(layer_idx=0)
        assert 0 in jac.shard_manifest
        assert len(jac.shard_manifest[0]) > 0

    def test_reload_manifest_raises_if_no_directory(self, jac):
        with pytest.raises(FileNotFoundError):
            jac._reload_manifest_from_disk(layer_idx=99)

    def test_reload_manifest_shard_order(self, jac):
        """Shards should be reloaded in correct numerical order, not lexicographic."""
        jac.compute()
        jac.shard_manifest.clear()
        jac._reload_manifest_from_disk(layer_idx=0)
        indices = [int(p.stem.split("_")[1]) for p in jac.shard_manifest[0]]
        assert indices == sorted(indices)