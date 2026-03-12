from __future__ import annotations
import dataclasses
import torch
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
import numpy as np
if TYPE_CHECKING:
    from KFAC import KFAC          # only for type hints — no hard dependency
    from Jacobian import Jacobian 

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
        power_iter_steps: int = 20,
        start_layer: int = 0
    ):
        self.mmodel = mmodel
        self.layer_inputs = layer_inputs
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size
        self.power_iter_steps = power_iter_steps

        self.layers = mmodel.model.model.layers[start_layer:]
        self.start_layer = start_layer
        self.seq_len = layer_inputs.hidden_states.shape[1]
        self.J_size = self.seq_len * mmodel.model.config.hidden_size

        # Tracks {layer_idx: [shard_paths]} after compute() is called
        self.shard_manifest: dict[int, list[Path]] = {}
        self.spectral_norms: dict[int, float] = {}
        self.converged_vectors: dict[int, torch.Tensor] = {}  # ← add this

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
    
    
    def _power_iteration_from_disk(
        self, 
        layer_idx: int, 
        steps: int | None = None,
        tol: float = 1e-4,
    ) -> float:
        steps = steps or self.power_iter_steps
        paths = self.shard_manifest[layer_idx]

        v = torch.randn(self.J_size)
        v = v / v.norm()
        sigma_prev = None
        sigma_curr = 0.0

        for step in range(steps):
            u_chunks = []
            for path in paths:
                J_chunk = torch.load(path, weights_only=True)
                u_chunks.append(J_chunk @ v)
                del J_chunk
            u = torch.cat(u_chunks, dim=0)
            sigma_curr = u.norm().item()   # ← estimate BEFORE normalizing
            u = u / u.norm()

            v = torch.zeros(self.J_size)
            offset = 0
            for path in paths:
                J_chunk = torch.load(path, weights_only=True)
                n_rows = J_chunk.shape[0]
                v += J_chunk.T @ u[offset: offset + n_rows]
                offset += n_rows
                del J_chunk
            v = v / v.norm()

            if sigma_prev is not None:
                relative_change = abs(sigma_curr - sigma_prev) / (sigma_prev + 1e-8)
                print(f"  Step {step}: σ₁ = {sigma_curr:.6f}, Δ = {relative_change:.2e}")
                if relative_change < tol:
                    print(f"  Converged at step {step}")
                    self.spectral_norms[layer_idx] = sigma_curr
                    self.converged_vectors[layer_idx] = v.clone()
                    return sigma_curr

            sigma_prev = sigma_curr

        print(f"  Warning: did not converge after {steps} steps, last σ₁ = {sigma_curr:.6f}")

        self.spectral_norms[layer_idx] = sigma_curr
        self.converged_vectors[layer_idx] = v.clone()

        return sigma_curr
    

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
        for l, _ in enumerate(tqdm(self.layers, desc="Layers"), start=self.start_layer):
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

            # self.spectral_norms[l] = self._power_iteration_from_disk(l)

            # thread hidden states forward to next layer
            with torch.no_grad():
                self.layer_inputs = self.mmodel.forward_layer(l, self.layer_inputs, no_grad=True)

    def compute_autograd(self, layer_idx: int) -> torch.Tensor:
        """
        Compute the full Jacobian for a layer using torch.autograd.functional.jacobian.
        Only feasible for small layers where the full matrix fits in memory.
        """
        h_in = self.layer_inputs.hidden_states.detach().requires_grad_(True)
        inputs_for_jacobian = dataclasses.replace(self.layer_inputs, hidden_states=h_in)

        def forward(h):
            updated_inputs = dataclasses.replace(inputs_for_jacobian, hidden_states=h)
            return (
                self.mmodel.forward_layer(layer_idx, updated_inputs, no_grad=False)
                .hidden_states[0]
                .reshape(-1)
            )

        J = torch.autograd.functional.jacobian(forward, h_in)
        return J.reshape(self.J_size, self.J_size).cpu()
    


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
        sigma = self._power_iteration_from_disk(layer_idx, steps)
        
        return sigma
    




class JacobianVisualizer:
    """
    Visualization companion for the Jacobian class.

    Parameters
    ----------
    jacobian : Jacobian
        A Jacobian instance after compute() has been called (so that
        spectral_norms and converged_vectors are populated).
    """

    # ── shared palette (mirrors KFACVisualizer dark theme) ────────────────
    _BG        = "#080c14"
    _FG        = "#e8e8e8"
    _ACCENT_A  = "#00e5ff"   # cyan  — used for Jacobian σ₁ line
    _ACCENT_B  = "#ff3cac"   # magenta — used for KFAC eigenvalue overlay
    _GRID      = "#ffffff"

    _SENS_COLORS = [
        "#080c14",   # black  (zero sensitivity)
        "#0d1f3c",
        "#1a3a6b",
        "#2e5fa3",
        "#4b8ec7",
        "#74bde0",
        "#a8d8ea",
        "#d4eef7",
        "#ffffff",   # white  (max sensitivity)
    ]
    SENSITIVITY_CMAP = LinearSegmentedColormap.from_list(
        "sensitivity", _SENS_COLORS, N=512
    )

    _SPEC_COLORS = [
        "#080c14",
        "#1a2744",
        "#2d3f6b",
        "#4b3f8c",
        "#8b3a9e",
        "#c94f4f",
        "#e8882a",
        "#f5d060",
        "#ffffff",
    ]
    SPECTRAL_CMAP = LinearSegmentedColormap.from_list(
        "spectral_bg", _SPEC_COLORS, N=512
    )

    def __init__(self, jacobian: "Jacobian"):
        self.jac = jacobian

    # ── helpers ───────────────────────────────────────────────────────────

    def _rc(self):
        """Apply shared rcParams for the dark monospace theme."""
        plt.rcParams.update({
            "font.family":      "monospace",
            "text.color":       self._FG,
            "axes.labelcolor":  self._FG,
            "xtick.color":      self._FG,
            "ytick.color":      self._FG,
            "axes.edgecolor":   "#2a3050",
            "figure.facecolor": self._BG,
            "axes.facecolor":   self._BG,
        })

    def _title_pe(self):
        return [patheffects.withStroke(linewidth=3, foreground=self._BG)]

    def _build_sensitivity_matrix(self) -> np.ndarray:
        """
        Returns [n_layers, seq_len] matrix of per-token sensitivity norms.
        For each layer l, takes the converged v vector (shape [J_size]),
        reshapes to [seq_len, hidden_size], and takes the L2 norm across
        the hidden dimension.
        """
        jac = self.jac
        if not jac.converged_vectors:
            raise RuntimeError(
                "converged_vectors is empty. Run compute() with the patched "
                "Jacobian, or call spectral_norm_from_disk() per layer."
            )

        n_layers = len(jac.converged_vectors)
        seq_len  = jac.seq_len
        hidden   = jac.J_size // seq_len

        mat = np.zeros((n_layers, seq_len))
        for l in sorted(jac.converged_vectors.keys()):
            v = jac.converged_vectors[l].float()        # [J_size]
            v_grid = v.reshape(seq_len, hidden)          # [seq_len, hidden]
            sensitivity = v_grid.norm(dim=-1).numpy()    # [seq_len]
            mat[l] = sensitivity / (sensitivity.max() + 1e-12)   # row-normalise

        return mat

    # ── public plots ──────────────────────────────────────────────────────

    def plot_spectral_profile(
        self,
        kfac: "KFAC | None" = None,
        proj_name: str = "mlp.down_proj",
        figsize: tuple = (13, 5),
        save_path: str | None = None,
    ):
        """
        Line plot of Jacobian σ₁ across layers.

        Parameters
        ----------
        kfac      : optional KFAC instance. If supplied, overlays the
                    max_eigenvalue for `proj_name` on a twin y-axis.
        proj_name : which KFAC projection to overlay (default: mlp.down_proj).
        """
        self._rc()

        jac = self.jac
        if not jac.spectral_norms:
            raise RuntimeError("spectral_norms is empty. Run compute() first.")

        layers      = sorted(jac.spectral_norms.keys())
        sigma_vals  = np.array([jac.spectral_norms[l] for l in layers])
        n_layers    = len(layers)

        fig, ax = plt.subplots(figsize=figsize, facecolor=self._BG)
        ax.set_facecolor(self._BG)

        # ── subtle gradient fill under σ₁ curve ──────────────────────────
        ax.fill_between(layers, sigma_vals, alpha=0.18, color=self._ACCENT_A, zorder=1)

        # ── σ₁ line ───────────────────────────────────────────────────────
        ax.plot(
            layers, sigma_vals,
            color=self._ACCENT_A, linewidth=2.2, zorder=3,
            label="Jacobian  σ₁  (activation stretch)",
            marker="o", markersize=4.5, markerfacecolor=self._BG,
            markeredgecolor=self._ACCENT_A, markeredgewidth=1.5,
        )

        # ── annotate peak ─────────────────────────────────────────────────
        peak_l = layers[int(np.argmax(sigma_vals))]
        peak_v = sigma_vals.max()
        ax.annotate(
            f"peak  L{peak_l}\nσ₁={peak_v:.3f}",
            xy=(peak_l, peak_v),
            xytext=(peak_l + max(1, n_layers * 0.05), peak_v * 0.92),
            fontsize=8, color=self._ACCENT_A, fontfamily="monospace",
            arrowprops=dict(arrowstyle="-|>", color=self._ACCENT_A,
                            lw=1.2, connectionstyle="arc3,rad=-0.2"),
            path_effects=self._title_pe(),
        )

        # ── KFAC overlay (twin axis) ──────────────────────────────────────
        if kfac is not None:
            kfac_vals = np.array([
                kfac.max_eigenvalues.get(l, {}).get(proj_name, np.nan)
                for l in layers
            ])
            valid = ~np.isnan(kfac_vals)

            ax2 = ax.twinx()
            ax2.set_facecolor(self._BG)
            ax2.tick_params(colors=self._ACCENT_B, labelsize=8)
            ax2.spines["right"].set_edgecolor(self._ACCENT_B)
            ax2.spines["left"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)

            ax2.plot(
                np.array(layers)[valid], kfac_vals[valid],
                color=self._ACCENT_B, linewidth=1.8, zorder=2,
                linestyle="--", alpha=0.85,
                label=f"KFAC  λ_max  ({proj_name})",
                marker="s", markersize=3.8, markerfacecolor=self._BG,
                markeredgecolor=self._ACCENT_B, markeredgewidth=1.4,
            )
            ax2.set_ylabel(
                f"KFAC  max eigenvalue  [{proj_name}]",
                fontsize=9, color=self._ACCENT_B, fontfamily="monospace", labelpad=10,
            )
            ax2.yaxis.set_tick_params(labelcolor=self._ACCENT_B)

            # combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2, labels1 + labels2,
                fontsize=8, facecolor="#0d1525", edgecolor="#2a3050",
                labelcolor=self._FG, loc="upper left",
            )
        else:
            ax.legend(
                fontsize=8, facecolor="#0d1525", edgecolor="#2a3050",
                labelcolor=self._FG, loc="upper left",
            )

        # ── grid & spines ─────────────────────────────────────────────────
        ax.grid(axis="y", color=self._GRID, alpha=0.07, linewidth=0.7, zorder=0)
        ax.grid(axis="x", color=self._GRID, alpha=0.04, linewidth=0.5, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_edgecolor("#2a3050")

        # ── labels ────────────────────────────────────────────────────────
        ax.set_xlabel("Layer", fontsize=11, fontfamily="monospace", labelpad=8)
        ax.set_ylabel("σ₁  (max singular value)", fontsize=11,
                      fontfamily="monospace", labelpad=8, color=self._ACCENT_A)
        ax.yaxis.set_tick_params(labelcolor=self._ACCENT_A)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7, rotation=45, ha="right")

        ax.set_title(
            "Jacobian Spectral Profile  ·  σ₁ per Transformer Layer",
            fontsize=12, pad=16, color=self._FG,
            fontfamily="monospace",
            path_effects=self._title_pe(),
        )

        fig.text(
            0.5, 0.01,
            "σ₁ > 1  →  layer amplifies representations  ·  "
            "σ₁ < 1  →  layer compresses representations",
            ha="center", fontsize=8, color="#666e80", fontfamily="monospace",
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved → {save_path}")

        plt.show()
        return fig, ax
    


    def plot_spectral_profile_simple(
        self,
        kfac: "KFAC | None" = None,
        proj_name: str = "mlp.down_proj",
        figsize: tuple = (13, 5),
        save_path: str | None = None,
    ):
        jac = self.jac
        if not jac.spectral_norms:
            raise RuntimeError("spectral_norms is empty. Run compute() first.")

        layers     = sorted(jac.spectral_norms.keys())
        sigma_vals = np.array([jac.spectral_norms[l] for l in layers])

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(layers, sigma_vals, marker="o", linewidth=1.8, label="Jacobian σ₁")

        if kfac is not None:
            kfac_vals = np.array([
                kfac.max_eigenvalues.get(l, {}).get(proj_name, np.nan)
                for l in layers
            ])
            valid = ~np.isnan(kfac_vals)
            ax2 = ax.twinx()
            ax2.plot(
                np.array(layers)[valid], kfac_vals[valid],
                marker="s", linewidth=1.8, linestyle="--",
                color="tomato", label=f"KFAC λ_max ({proj_name})",
            )
            ax2.set_ylabel(f"KFAC max eigenvalue [{proj_name}]")
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2)
        else:
            ax.legend()

        ax.set_xlabel("Layer")
        ax.set_ylabel("σ₁ (max singular value)")
        ax.set_title("Jacobian Spectral Profile — σ₁ per Layer")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {save_path}")

        plt.show()
        return fig, ax



    def plot_correlation_scatter(
        self,
        kfac: "KFAC",
        figsize: tuple = (9, 6),
        save_path: str | None = None,
    ):
        PROJ_ORDER  = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        PROJ_LABELS = ["Q", "K", "V", "O", "Gate", "Up", "Down"]

        jac    = self.jac
        layers = sorted(jac.spectral_norms.keys())

        fig, ax = plt.subplots(figsize=figsize)

        for proj, label in zip(PROJ_ORDER, PROJ_LABELS):
            sigma_vals = []
            kfac_vals  = []

            for l in layers:
                sigma    = jac.spectral_norms.get(l)
                kfac_eig = kfac.max_eigenvalues.get(l, {}).get(proj)
                if sigma is not None and kfac_eig is not None:
                    sigma_vals.append(sigma)
                    kfac_vals.append(kfac_eig)

            if not sigma_vals:
                continue

            sigma_vals = np.array(sigma_vals)
            kfac_vals  = np.array(kfac_vals)

            ax.scatter(sigma_vals, kfac_vals, label=label, s=35, zorder=3)

            # fit in log space
            m, b = np.polyfit(np.log10(sigma_vals), np.log10(kfac_vals), 1)
            x_line = np.linspace(sigma_vals.min(), sigma_vals.max(), 200)
            y_line = 10 ** (m * np.log10(x_line) + b)
            ax.plot(x_line, y_line, linewidth=1.0, alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Jacobian σ₁  (log scale)")
        ax.set_ylabel("KFAC max eigenvalue  (log scale)")
        ax.set_title("Jacobian σ₁  vs  KFAC λ_max  —  all projections")
        ax.legend(title="Projection", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {save_path}")

        plt.show()
        return fig, ax



    def plot_sensitivity_heatmap(
        self,
        token_labels: list[str] | None = None,
        figsize: tuple = (14, 9),
        save_path: str | None = None,
    ):
        """
        Heatmap of per-token input sensitivity across layers.

        Rows   = layers (0 at top)
        Cols   = token positions
        Color  = L2 norm of v reshaped to [seq_len, hidden_size],
                 row-normalised per layer so each layer uses the full
                 colour range independently.

        Parameters
        ----------
        token_labels : optional list of decoded token strings (length = seq_len).
                       Falls back to raw indices if None.
        """
        self._rc()

        mat     = self._build_sensitivity_matrix()   # [n_layers, seq_len]
        n_layers, seq_len = mat.shape

        if token_labels is not None and len(token_labels) != seq_len:
            raise ValueError(
                f"token_labels has {len(token_labels)} entries but seq_len={seq_len}."
            )

        x_labels = token_labels if token_labels is not None else [str(i) for i in range(seq_len)]

        fig, ax = plt.subplots(figsize=figsize, facecolor=self._BG)
        ax.set_facecolor(self._BG)

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=self.SENSITIVITY_CMAP,
            vmin=0.0, vmax=1.0,
            interpolation="nearest",
            extent=[-0.5, seq_len - 0.5, n_layers - 0.5, -0.5],
        )

        # ── token x-axis ──────────────────────────────────────────────────
        # Show all labels if seq_len ≤ 64, otherwise thin to ~32 ticks
        if seq_len <= 64:
            tick_positions = np.arange(seq_len)
            tick_labels    = x_labels
        else:
            step = max(1, seq_len // 32)
            tick_positions = np.arange(0, seq_len, step)
            tick_labels    = [x_labels[i] for i in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels, fontsize=7, rotation=55, ha="right",
            fontfamily="monospace",
        )

        # ── layer y-axis ───────────────────────────────────────────────────
        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels(
            [f"L{l:02d}" for l in range(n_layers)],
            fontsize=7.5, fontfamily="monospace",
        )

        # ── subtle grid ────────────────────────────────────────────────────
        for x in np.arange(-0.5, seq_len, 1):
            ax.axvline(x, color=self._GRID, lw=0.15, alpha=0.10, zorder=1)
        for y in np.arange(-0.5, n_layers, 1):
            ax.axhline(y, color=self._GRID, lw=0.25, alpha=0.18, zorder=1)

        # ── colourbar ─────────────────────────────────────────────────────
        cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, aspect=28)
        cbar.set_label(
            "Input sensitivity  (row-normalised per layer)",
            fontsize=9, color=self._FG, fontfamily="monospace",
        )
        cbar.ax.yaxis.set_tick_params(color=self._FG)
        plt.setp(cbar.ax.yaxis.get_ticklabels(),
                 color=self._FG, fontfamily="monospace", fontsize=8)
        cbar.outline.set_edgecolor("#444")

        # ── labels & title ─────────────────────────────────────────────────
        ax.set_xlabel("Token position", fontsize=11,
                      fontfamily="monospace", labelpad=8)
        ax.set_ylabel("Layer", fontsize=11,
                      fontfamily="monospace", labelpad=8)
        ax.set_title(
            "Input Sensitivity Heatmap  ·  ‖v[token]‖  per Layer",
            fontsize=12, pad=16, color=self._FG,
            fontfamily="monospace",
            path_effects=self._title_pe(),
        )

        fig.text(
            0.5, 0.01,
            "Bright  →  token position dominates the layer's largest singular direction  ·  "
            "Dark  →  position contributes little",
            ha="center", fontsize=8, color="#666e80", fontfamily="monospace",
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved → {save_path}")

        plt.show()
        return fig, ax

    def plot_all(
        self,
        kfac: "KFAC | None" = None,
        proj_name: str = "mlp.down_proj",
        token_labels: list[str] | None = None,
        save_path: str | None = None,
    ):
        """
        Renders both plots stacked in one figure.
        """
        self._rc()
        fig = plt.figure(figsize=(14, 16), facecolor=self._BG)
        gs  = fig.add_gridspec(2, 1, height_ratios=[1, 1.6], hspace=0.45)

        ax_spec = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])

        # delegate to the individual methods but inject axes
        # (re-implement inline so we can share the figure)
        self._plot_spectral_into(ax_spec, kfac=kfac, proj_name=proj_name)
        self._plot_heatmap_into(ax_heat, token_labels=token_labels)

        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved → {save_path}")

        plt.show()
        return fig

    # ── private axis-injection helpers for plot_all ───────────────────────

    def _plot_spectral_into(self, ax, kfac=None, proj_name="mlp.down_proj"):
        jac        = self.jac
        layers     = sorted(jac.spectral_norms.keys())
        sigma_vals = np.array([jac.spectral_norms[l] for l in layers])
        n_layers   = len(layers)

        ax.set_facecolor(self._BG)
        ax.fill_between(layers, sigma_vals, alpha=0.18, color=self._ACCENT_A, zorder=1)
        ax.plot(
            layers, sigma_vals,
            color=self._ACCENT_A, linewidth=2.2, zorder=3,
            label="Jacobian  σ₁",
            marker="o", markersize=4.5, markerfacecolor=self._BG,
            markeredgecolor=self._ACCENT_A, markeredgewidth=1.5,
        )

        peak_l = layers[int(np.argmax(sigma_vals))]
        peak_v = sigma_vals.max()
        ax.annotate(
            f"peak  L{peak_l}  σ₁={peak_v:.3f}",
            xy=(peak_l, peak_v),
            xytext=(peak_l + max(1, n_layers * 0.05), peak_v * 0.90),
            fontsize=7.5, color=self._ACCENT_A, fontfamily="monospace",
            arrowprops=dict(arrowstyle="-|>", color=self._ACCENT_A, lw=1.1,
                            connectionstyle="arc3,rad=-0.2"),
        )

        if kfac is not None:
            kfac_vals = np.array([
                kfac.max_eigenvalues.get(l, {}).get(proj_name, np.nan)
                for l in layers
            ])
            valid = ~np.isnan(kfac_vals)
            ax2 = ax.twinx()
            ax2.set_facecolor(self._BG)
            ax2.tick_params(colors=self._ACCENT_B, labelsize=7)
            ax2.spines["right"].set_edgecolor(self._ACCENT_B)
            for s in ["left", "top", "bottom"]:
                ax2.spines[s].set_visible(False)
            ax2.plot(
                np.array(layers)[valid], kfac_vals[valid],
                color=self._ACCENT_B, linewidth=1.8, linestyle="--",
                alpha=0.85, label=f"KFAC  λ_max  ({proj_name})",
                marker="s", markersize=3.5, markerfacecolor=self._BG,
                markeredgecolor=self._ACCENT_B, markeredgewidth=1.3, zorder=2,
            )
            ax2.set_ylabel(f"KFAC  λ_max  [{proj_name}]",
                           fontsize=8, color=self._ACCENT_B,
                           fontfamily="monospace", labelpad=8)
            ax2.yaxis.set_tick_params(labelcolor=self._ACCENT_B)
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=7.5,
                      facecolor="#0d1525", edgecolor="#2a3050", labelcolor=self._FG)
        else:
            ax.legend(fontsize=7.5, facecolor="#0d1525",
                      edgecolor="#2a3050", labelcolor=self._FG)

        ax.grid(axis="y", color=self._GRID, alpha=0.07, lw=0.7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_edgecolor("#2a3050")

        ax.set_xlabel("Layer", fontsize=10, fontfamily="monospace")
        ax.set_ylabel("σ₁", fontsize=10, fontfamily="monospace",
                      color=self._ACCENT_A)
        ax.yaxis.set_tick_params(labelcolor=self._ACCENT_A)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=6.5,
                           rotation=45, ha="right")
        ax.set_title("Jacobian Spectral Profile", fontsize=11,
                     color=self._FG, fontfamily="monospace", pad=10,
                     path_effects=self._title_pe())

    def _plot_heatmap_into(self, ax, token_labels=None):
        mat = self._build_sensitivity_matrix()
        n_layers, seq_len = mat.shape

        x_labels = token_labels if token_labels is not None else [str(i) for i in range(seq_len)]

        ax.set_facecolor(self._BG)
        im = ax.imshow(
            mat, aspect="auto", cmap=self.SENSITIVITY_CMAP,
            vmin=0.0, vmax=1.0, interpolation="nearest",
            extent=[-0.5, seq_len - 0.5, n_layers - 0.5, -0.5],
        )

        step = max(1, seq_len // 32) if seq_len > 64 else 1
        ticks = np.arange(0, seq_len, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([x_labels[i] for i in ticks],
                           fontsize=7, rotation=55, ha="right",
                           fontfamily="monospace")
        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels([f"L{l:02d}" for l in range(n_layers)],
                           fontsize=7.5, fontfamily="monospace")

        for x in np.arange(-0.5, seq_len, 1):
            ax.axvline(x, color=self._GRID, lw=0.15, alpha=0.10)
        for y in np.arange(-0.5, n_layers, 1):
            ax.axhline(y, color=self._GRID, lw=0.25, alpha=0.18)

        cbar = ax.get_figure().colorbar(im, ax=ax, shrink=0.55, pad=0.02, aspect=28)
        cbar.set_label("sensitivity  (row-norm)", fontsize=8,
                       color=self._FG, fontfamily="monospace")
        cbar.ax.yaxis.set_tick_params(color=self._FG)
        plt.setp(cbar.ax.yaxis.get_ticklabels(),
                 color=self._FG, fontfamily="monospace", fontsize=7.5)
        cbar.outline.set_edgecolor("#444")

        ax.set_xlabel("Token position", fontsize=10, fontfamily="monospace")
        ax.set_ylabel("Layer", fontsize=10, fontfamily="monospace")
        ax.set_title("Input Sensitivity Heatmap  ·  ‖v[token]‖ per Layer",
                     fontsize=11, color=self._FG, fontfamily="monospace", pad=10,
                     path_effects=self._title_pe())
    