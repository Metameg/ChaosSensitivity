# import torch
# import torch.nn as nn
# import pytest
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from main import KFAC, LayerInputs, MyModel


# # ─────────────────────────────────────────────
# # Test 1: Factor shapes
# # ─────────────────────────────────────────────

# def test_factor_shapes(kfac: KFAC, expected_dims: dict):
#     """
#     Verify A is (d_in, d_in) and G is (d_out, d_out) for each linear layer.
#     expected_dims: {linear_name: (d_in, d_out)}
#     e.g. {"self_attn.q_proj": (4096, 1024), "mlp.down_proj": (14336, 4096)}
#     """
#     for l, layer_factors in kfac.factors.items():
#         for name, (A, G) in layer_factors.items():
#             if name in expected_dims:
#                 d_in, d_out = expected_dims[name]
#                 assert A.shape == (d_in, d_in), (
#                     f"Layer {l} {name}: expected A shape ({d_in}, {d_in}), got {A.shape}"
#                 )
#                 assert G.shape == (d_out, d_out), (
#                     f"Layer {l} {name}: expected G shape ({d_out}, {d_out}), got {G.shape}"
#                 )
#     print("PASSED: test_factor_shapes")


# # ─────────────────────────────────────────────
# # Test 2: A and G are symmetric positive semidefinite
# # ─────────────────────────────────────────────

# def test_factors_spsd(kfac: KFAC, tol: float = 1e-4):
#     """
#     Verify A and G are symmetric and have no significantly negative eigenvalues.
#     """
#     for l, layer_factors in kfac.factors.items():
#         for name, (A, G) in layer_factors.items():
#             # Symmetry
#             assert torch.allclose(A, A.T, atol=tol), (
#                 f"Layer {l} {name}: A is not symmetric"
#             )
#             assert torch.allclose(G, G.T, atol=tol), (
#                 f"Layer {l} {name}: G is not symmetric"
#             )

#             # No significantly negative eigenvalues
#             eigs_A = torch.linalg.eigvalsh(A.double())
#             eigs_G = torch.linalg.eigvalsh(G.double())
#             assert eigs_A.min().item() >= -tol, (
#                 f"Layer {l} {name}: A has negative eigenvalue {eigs_A.min().item():.6f}"
#             )
#             assert eigs_G.min().item() >= -tol, (
#                 f"Layer {l} {name}: G has negative eigenvalue {eigs_G.min().item():.6f}"
#             )
#     print("PASSED: test_factors_spsd")


# # ─────────────────────────────────────────────
# # Test 3: Exact reconstruction at layer 0
# # ─────────────────────────────────────────────

# def test_reconstruction_layer_0(kfac: KFAC, layer_inputs: LayerInputs, tol: float = 1e-4):
#     """
#     Verify that g.T @ a exactly matches module.weight.grad at layer 0.
#     """
#     kfac.mmodel.model.zero_grad()
#     layer = kfac.layers[0]

#     activations, gradients = KFAC.compute_layer_gradients(
#         kfac.mmodel, layer_inputs, 0, kfac.target_token_id, kfac.layers
#     )

#     results = KFAC.verify_reconstruction(layer, activations, gradients)

#     for name, diff in results.items():
#         if diff is not None:
#             max_diff, mean_diff = diff
#             assert max_diff < tol, (
#                 f"Layer 0 {name}: reconstruction max_diff {max_diff:.6f} exceeds tolerance {tol}"
#             )
#         else:
#             print(f"  Layer 0 {name}: no .grad found, skipping")

#     print("PASSED: test_reconstruction_layer_0")


# # ─────────────────────────────────────────────
# # Test 4: Eigenvalue ordering across layers
# # ─────────────────────────────────────────────

# def test_eigenvalue_layer_trend(kfac: KFAC, projection: str = "mlp.down_proj"):
#     """
#     Plots and checks that max eigenvalues across layers show a meaningful
#     distribution rather than being all identical (which would indicate a bug).
#     """
#     eigenvalues_per_layer = []

#     for l in sorted(kfac.max_eigenvalues.keys()):
#         if projection in kfac.max_eigenvalues[l]:
#             eigenvalues_per_layer.append((l, kfac.max_eigenvalues[l][projection]))

#     values = [v for _, v in eigenvalues_per_layer]

#     # Check they are not all identical
#     assert max(values) != min(values), (
#         f"Max eigenvalues for {projection} are identical across all layers — likely a bug"
#     )

#     # Print for visual inspection
#     print(f"\nMax eigenvalue trend for {projection}:")
#     for l, v in eigenvalues_per_layer:
#         print(f"  Layer {l:2d}: {v:.4f}")

#     print(f"\nMin: {min(values):.4f}  Max: {max(values):.4f}  Ratio: {max(values)/min(values):.2f}x")
#     print("PASSED: test_eigenvalue_layer_trend")


# # ─────────────────────────────────────────────
# # Test 5: Eigenvalue ordering across projections within a layer
# # ─────────────────────────────────────────────

# def test_eigenvalue_projection_diversity(kfac: KFAC, layer_idx: int = 0):
#     """
#     Verify that different projections within the same layer have
#     meaningfully different max eigenvalues.
#     """
#     layer_eigs = kfac.max_eigenvalues[layer_idx]

#     print(f"\nMax eigenvalues for layer {layer_idx}:")
#     values = []
#     for name, val in layer_eigs.items():
#         print(f"  {name:40s}: {val:.4f}")
#         values.append(val)

#     assert max(values) != min(values), (
#         f"All projections in layer {layer_idx} have identical max eigenvalues — likely a bug"
#     )

#     print("PASSED: test_eigenvalue_projection_diversity")


# # ─────────────────────────────────────────────
# # Test 6: Eigenvalue scaling with sequence length
# # ─────────────────────────────────────────────

# def test_eigenvalue_scaling_with_seq_len(mmodel, prompt: str, target_token_id: int, projection: str = "self_attn.q_proj", layer_idx: int = 0):
#     """
#     Verify that max eigenvalues grow roughly proportionally with sequence length T.
#     Runs KFAC on the same prompt truncated to different lengths and checks scaling.
#     """
#     layers = mmodel.model.model.layers
#     results = {}

#     for seq_len in [10, 20, 30]:
#         input_ids = mmodel.tokenize(prompt)
#         input_ids = input_ids[:, :seq_len]

#         if input_ids.shape[1] < seq_len:
#             print(f"  Prompt too short for seq_len={seq_len}, skipping")
#             continue

#         layer_inputs = mmodel.prepare_layer_inputs(input_ids)
#         kfac = KFAC(mmodel, layers, target_token_id)
#         kfac.collect_factors(layer_inputs)
#         kfac.compute_eigenvalues()

#         if projection in kfac.max_eigenvalues[layer_idx]:
#             results[seq_len] = kfac.max_eigenvalues[layer_idx][projection]

#     print(f"\nEigenvalue scaling for {projection} at layer {layer_idx}:")
#     seq_lens = sorted(results.keys())
#     for t in seq_lens:
#         print(f"  T={t:3d}: {results[t]:.4f}")

#     # Check that eigenvalues grow with sequence length (not necessarily linearly,
#     # but should be monotonically increasing)
#     values = [results[t] for t in seq_lens]
#     assert values == sorted(values), (
#         f"Max eigenvalues did not increase monotonically with sequence length: {results}"
#     )

#     print("PASSED: test_eigenvalue_scaling_with_seq_len")


# # ─────────────────────────────────────────────
# # Test 7: Perturbation test
# # ─────────────────────────────────────────────

# def test_perturbation(kfac: KFAC, layer_inputs: LayerInputs, layer_idx: int = 0, projection: str = "self_attn.q_proj", delta: float = 1e-5):
#     """
#     Perturbs the top eigenvector direction of a weight matrix and checks
#     that the change in loss matches 0.5 * delta^2 * max_eigenvalue.
#     """
#     # Find the target linear module
#     target_module = None
#     for name, module in kfac.layers[layer_idx].named_modules():
#         if name == projection and isinstance(module, nn.Linear):
#             target_module = module
#             break

#     assert target_module is not None, f"Could not find {projection} in layer {layer_idx}"
    
#     A, G = kfac.factors[layer_idx][projection]
#     # Get the top eigenvector of the weight gradient direction
#     # top eigenvector of A
#     _, vecs_A = torch.linalg.eigh(A.double())
#     top_vec_A = vecs_A[:, -1].float()

#     # top eigenvector of G
#     _, vecs_G = torch.linalg.eigh(G.double())
#     top_vec_G = vecs_G[:, -1].float()

#     # top eigenvector of A ⊗ G is the outer product, reshaped to weight shape
#     top_direction = torch.outer(top_vec_G, top_vec_A)  # (d_out, d_in)

#     # Compute baseline loss
#     def compute_loss(mmodel, layer_inputs, kfac, layer_idx):
#         h = layer_inputs.hidden_states.detach()
#         current_inputs = layer_inputs
#         for i in range(layer_idx, len(kfac.layers)):
#             out = mmodel.forward_layer(i, current_inputs, no_grad=False)
#             current_inputs = out
#         logits = mmodel.lm_head(mmodel.norm(current_inputs.hidden_states))
#         log_prob = torch.log_softmax(logits[0, -1, :], dim=-1)
#         return log_prob[kfac.target_token_id].item()

#     with torch.no_grad():
#         loss_base = compute_loss(kfac.mmodel, layer_inputs, kfac, layer_idx)

#         # Perturb in top eigenvector direction
#         target_module.weight.data += delta * top_direction.to(target_module.weight.device)
#         loss_perturbed = compute_loss(kfac.mmodel, layer_inputs, kfac, layer_idx)

#         # Restore weight
#         target_module.weight.data -= delta * top_direction.to(target_module.weight.device)

#     actual_change = abs(loss_perturbed - loss_base)
#     max_eig = kfac.max_eigenvalues[layer_idx][projection]
#     predicted_change = 0.5 * delta ** 2 * max_eig

#     ratio = actual_change / predicted_change if predicted_change > 0 else float('inf')

#     print(f"\nPerturbation test for {projection} at layer {layer_idx}:")
#     print(f"  Actual loss change:    {actual_change:.6f}")
#     print(f"  Predicted (KFAC):      {predicted_change:.6f}")
#     print(f"  Ratio actual/predicted: {ratio:.4f}")

#     # KFAC is an approximation so we allow a loose tolerance
#     if 0.01 < ratio < 100:
#         print(f"Perturbation ratio {ratio:.4f} is outside acceptable range — KFAC estimate may be wrong")
    

#     print("PASSED: test_perturbation")




# # ─────────────────────────────────────────────
# # Runner
# # ─────────────────────────────────────────────

# if __name__ == "__main__":
    

#     prompt = "The quick brown fox"
#     mmodel = MyModel()
#     mmodel.model.config._attn_implementation = "eager"

#     input_ids = mmodel.tokenize(prompt)
#     target_token_id = mmodel.tokenizer.encode(" negative", add_special_tokens=False)
#     layer_inputs = mmodel.prepare_layer_inputs(input_ids)
#     layers = mmodel.model.model.layers

#     # Llama-8B expected dims for spot checking
#     expected_dims = {
#         "self_attn.q_proj": (4096, 4096),
#         "self_attn.k_proj": (4096, 1024),
#         "self_attn.v_proj": (4096, 1024),
#         "self_attn.o_proj": (4096, 4096),
#         "mlp.gate_proj":    (4096, 14336),
#         "mlp.up_proj":      (4096, 14336),
#         "mlp.down_proj":    (14336, 4096),
#     }

#     kfac = KFAC(mmodel, layers, target_token_id)
#     kfac.run(layer_inputs)

#     test_factor_shapes(kfac, expected_dims)
#     test_factors_spsd(kfac)
#     test_reconstruction_layer_0(kfac, layer_inputs)
#     test_eigenvalue_layer_trend(kfac)
#     test_eigenvalue_projection_diversity(kfac)
#     test_eigenvalue_scaling_with_seq_len(mmodel, prompt, target_token_id)
#     test_perturbation(kfac, layer_inputs)
    


import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import KFAC, LayerInputs, MyModel


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def mmodel():
    m = MyModel()
    m.model.config._attn_implementation = "eager"
    return m


@pytest.fixture(scope="session")
def prompt():
    return "The quick brown fox"


@pytest.fixture(scope="session")
def target_token_id(mmodel):
    return mmodel.tokenizer.encode(" negative", add_special_tokens=False)


@pytest.fixture(scope="session")
def input_ids(mmodel, prompt):
    return mmodel.tokenize(prompt)


@pytest.fixture(scope="session")
def layer_inputs(mmodel, input_ids):
    return mmodel.prepare_layer_inputs(input_ids)


@pytest.fixture(scope="session")
def layers(mmodel):
    return mmodel.model.model.layers


@pytest.fixture(scope="session")
def kfac(mmodel, layers, target_token_id, layer_inputs):
    k = KFAC(mmodel, layers, target_token_id)
    k.collect_factors(layer_inputs) 
    k.compute_eigenvalues()          
    return k


@pytest.fixture
def expected_dims():
    return {
        "self_attn.q_proj": (4096, 4096),
        "self_attn.k_proj": (4096, 1024),
        "self_attn.v_proj": (4096, 1024),
        "self_attn.o_proj": (4096, 4096),
        "mlp.gate_proj":    (4096, 14336),
        "mlp.up_proj":      (4096, 14336),
        "mlp.down_proj":    (14336, 4096),
    }


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestFactorShapes:
    def test_factor_shapes(self, kfac, expected_dims):
        """A is (d_in, d_in) and G is (d_out, d_out) for each linear layer."""
        for l, layer_factors in kfac.factors.items():
            for name, (A, G) in layer_factors.items():
                if name in expected_dims:
                    d_in, d_out = expected_dims[name]
                    assert A.shape == (d_in, d_in), (
                        f"Layer {l} {name}: expected A shape ({d_in}, {d_in}), got {A.shape}"
                    )
                    assert G.shape == (d_out, d_out), (
                        f"Layer {l} {name}: expected G shape ({d_out}, {d_out}), got {G.shape}"
                    )


class TestFactorProperties:
    def test_factors_symmetric(self, kfac, tol=1e-4):
        """A and G should be symmetric."""
        for l, layer_factors in kfac.factors.items():
            for name, (A, G) in layer_factors.items():
                assert torch.allclose(A, A.T, atol=tol), f"Layer {l} {name}: A is not symmetric"
                assert torch.allclose(G, G.T, atol=tol), f"Layer {l} {name}: G is not symmetric"

    def test_factors_positive_semidefinite(self, kfac, tol=1e-4):
        """A and G should have no significantly negative eigenvalues."""
        for l, layer_factors in kfac.factors.items():
            for name, (A, G) in layer_factors.items():
                eigs_A = torch.linalg.eigvalsh(A.double())
                eigs_G = torch.linalg.eigvalsh(G.double())
                assert eigs_A.min().item() >= -tol, (
                    f"Layer {l} {name}: A has negative eigenvalue {eigs_A.min().item():.6f}"
                )
                assert eigs_G.min().item() >= -tol, (
                    f"Layer {l} {name}: G has negative eigenvalue {eigs_G.min().item():.6f}"
                )


class TestReconstruction:
    def test_reconstruction_layer_0(self, kfac, layer_inputs, tol=1e-4):
        """g.T @ a should exactly match module.weight.grad at layer 0."""
        kfac.mmodel.model.zero_grad()

        activations, gradients = KFAC.compute_layer_gradients(
            kfac.mmodel, layer_inputs, 0, kfac.target_token_id, kfac.layers
        )
        results = KFAC.verify_reconstruction(kfac.layers[0], activations, gradients)

        for name, diff in results.items():
            if diff is not None:
                max_diff, mean_diff = diff
                assert max_diff < tol, (
                    f"Layer 0 {name}: reconstruction max_diff {max_diff:.6f} exceeds tolerance {tol}"
                )


class TestEigenvalues:
    def test_eigenvalue_layer_trend(self, kfac, projection="mlp.down_proj"):
        """Max eigenvalues should vary across layers, not be identical."""
        values = [
            kfac.max_eigenvalues[l][projection]
            for l in sorted(kfac.max_eigenvalues.keys())
            if projection in kfac.max_eigenvalues[l]
        ]
        assert max(values) != min(values), (
            f"Max eigenvalues for {projection} are identical across all layers — likely a bug"
        )

    def test_eigenvalue_projection_diversity(self, kfac, layer_idx=0):
        """Different projections within the same layer should have different max eigenvalues."""
        values = list(kfac.max_eigenvalues[layer_idx].values())
        assert max(values) != min(values), (
            f"All projections in layer {layer_idx} have identical max eigenvalues — likely a bug"
        )

    def test_eigenvalue_scaling_with_seq_len(self, mmodel, prompt, target_token_id, projection="self_attn.q_proj", layer_idx=0):
        """Max eigenvalues should increase monotonically with sequence length."""
        layers = mmodel.model.model.layers
        results = {}

        for seq_len in [10, 20, 30]:
            input_ids = mmodel.tokenize(prompt)[:, :seq_len]
            if input_ids.shape[1] < seq_len:
                continue
            layer_inputs = mmodel.prepare_layer_inputs(input_ids)
            k = KFAC(mmodel, layers, target_token_id)
            k.collect_factors(layer_inputs)
            k.compute_eigenvalues()
            if projection in k.max_eigenvalues[layer_idx]:
                results[seq_len] = k.max_eigenvalues[layer_idx][projection]

        values = [results[t] for t in sorted(results.keys())]
        assert values == sorted(values), (
            f"Max eigenvalues did not increase monotonically with sequence length: {results}"
        )


class TestPerturbation:
    def test_perturbation(self, kfac, layer_inputs, layer_idx=0, projection="self_attn.q_proj", delta=1e-5):
        target_module = next(
            (m for name, m in kfac.layers[layer_idx].named_modules()
            if name == projection and isinstance(m, nn.Linear)),
            None
        )
        assert target_module is not None, f"Could not find {projection} in layer {layer_idx}"

        # top_eigenvectors stores outer(top_vec_G, top_vec_A) already flattened
        top_vec = kfac.top_eigenvectors[layer_idx][projection]
        top_direction = top_vec.reshape(target_module.weight.shape)

        def compute_loss(layer_inputs):
            current_inputs = layer_inputs
            for i in range(layer_idx, len(kfac.layers)):
                current_inputs = kfac.mmodel.forward_layer(i, current_inputs, no_grad=False)
            logits = kfac.mmodel.lm_head(kfac.mmodel.norm(current_inputs.hidden_states))
            return torch.log_softmax(logits[0, -1, :], dim=-1)[kfac.target_token_id].item()

        with torch.no_grad():
            loss_base = compute_loss(layer_inputs)
            target_module.weight.data += delta * top_direction.to(target_module.weight.device)
            loss_perturbed = compute_loss(layer_inputs)
            target_module.weight.data -= delta * top_direction.to(target_module.weight.device)

        actual_change = abs(loss_perturbed - loss_base)
        predicted_change = 0.5 * delta ** 2 * kfac.max_eigenvalues[layer_idx][projection]
        ratio = actual_change / predicted_change if predicted_change > 0 else float("inf")

        assert 0.01 < ratio < 100, (
            f"Perturbation ratio {ratio:.4f} outside acceptable range — KFAC estimate may be wrong"
        )

        def compute_loss(layer_inputs):
            current_inputs = layer_inputs
            for i in range(layer_idx, len(kfac.layers)):
                current_inputs = kfac.mmodel.forward_layer(i, current_inputs, no_grad=False)
            logits = kfac.mmodel.lm_head(kfac.mmodel.norm(current_inputs.hidden_states))
            return torch.log_softmax(logits[0, -1, :], dim=-1)[kfac.target_token_id].item()

        with torch.no_grad():
            loss_base = compute_loss(layer_inputs)
            target_module.weight.data += delta * top_direction.to(target_module.weight.device)
            loss_perturbed = compute_loss(layer_inputs)
            target_module.weight.data -= delta * top_direction.to(target_module.weight.device)

        actual_change = abs(loss_perturbed - loss_base)
        predicted_change = 0.5 * delta ** 2 * kfac.max_eigenvalues[layer_idx][projection]
        ratio = actual_change / predicted_change if predicted_change > 0 else float("inf")

        assert 0.01 < ratio < 100, (
            f"Perturbation ratio {ratio:.4f} is outside acceptable range — KFAC estimate may be wrong"
        )