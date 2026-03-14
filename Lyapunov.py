import torch
from data_types import LyapunovResult
from pathlib import Path
from typing import Optional
from Jacobian import Jacobian
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import Optional
from datetime import datetime

class Lyapunov:
    """
    Computes Lyapunov exponents of the composed per-layer Jacobians
    using the Benettin (QR) algorithm.
 
    Streams Jacobian shards from disk — the full composed Jacobian
    is never materialized in memory.
 
    Usage:
        jac = Jacobian(mmodel, layer_inputs, save_dir="jacobians")
        jac.compute()
 
        lyap = Lyapunov(jac)
        result = lyap.compute(k=1)
 
        print(result.exponents)           # [λ₁, ..., λ_k]
        print(result.per_layer_log_stretches)  # log stretch per layer
        print(result.lyapunov_vector.shape)    # [J_size]
    """
 
    def __init__(self, jacobian: Jacobian, base_save_dir: Path = Path("data/lyapunov")):
        """
        Args:
            jacobian: A Jacobian instance that has already had compute() called,
                      so shard_manifest is populated.
        """
        self.jacobian = jacobian
        self.J_size = jacobian.J_size
        self.result: Optional[LyapunovResult] = None
        self.BASE_SAVE_DIR = base_save_dir

    def _save_layer(self, layer_idx: int, Q: torch.Tensor, R_diag: torch.Tensor,
                log_stretches: torch.Tensor, log_stretch_sum: torch.Tensor) -> None:
        layer_dir = self.save_dir / f"layer_{layer_idx:04d}"
        layer_dir.mkdir(exist_ok=True)
        torch.save(Q.cpu(),               layer_dir / "Q.pt")
        torch.save(R_diag.cpu(),          layer_dir / "R_diag.pt")
        torch.save(log_stretches.cpu(),   layer_dir / "log_stretches.pt")
        torch.save(log_stretch_sum.cpu(), layer_dir / "log_stretch_sum.pt")

    def _save_manifest(self, layer_indices, per_layer_log_stretches,
                    exponents, lyapunov_vector, k) -> None:
        torch.save({
            "layer_indices":           layer_indices,
            "per_layer_log_stretches": per_layer_log_stretches,
            "exponents":               exponents,
            "lyapunov_vector":         lyapunov_vector.cpu(),
            "k":                       k,
            "J_size":                  self.J_size,
            "timestamp":               self.save_dir.name,
        }, self.save_dir / "manifest.pt")


 
    def _apply_jacobian(self, layer_idx: int, Q: torch.Tensor) -> torch.Tensor:
        """
        Computes Z = J_l @ Q by streaming shards from disk.
 
        Args:
            layer_idx: which layer's shards to load
            Q: tensor of shape [J_size, k]
 
        Returns:
            Z: tensor of shape [J_size, k]
        """
        paths = self.jacobian.shard_manifest[layer_idx]
        Z = torch.zeros(self.J_size, Q.shape[1])
        row_offset = 0
 
        for path in paths:
            J_chunk = torch.load(path, weights_only=True)   # [chunk_rows, J_size]
            n_rows = J_chunk.shape[0]
            Z[row_offset: row_offset + n_rows] = J_chunk.float() @ Q
            row_offset += n_rows
            del J_chunk
 
        return Z
    
    
 
    def compute(self, k: int = 1) -> LyapunovResult:
        """
        Run the Benettin algorithm over all layers in shard_manifest order.
 
        Args:
            k: number of Lyapunov exponents to compute.
               k=1 gives only the top exponent cheaply.
               Increasing k gives the full spectrum but memory scales as J_size * k.
 
        Returns:
            LyapunovResult dataclass.
        """
        self.save_dir = self.BASE_SAVE_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        layer_indices = sorted(self.jacobian.shard_manifest.keys())
        n_layers = len(layer_indices)
        k = min(k, self.J_size)
 
        # Initialize Q as a random orthonormal [J_size, k] matrix.
        # Almost any starting direction works — the top singular direction
        # of the composed Jacobian dominates after a few layers.
        Q = torch.randn(self.J_size, k)
        Q, _ = torch.linalg.qr(Q)
 
        log_stretch_sum = torch.zeros(k)
        per_layer_log_stretches = []
 
        for l in tqdm(layer_indices, desc="Lyapunov (Benettin)", unit="layer", ncols=80):
            # Apply this layer's Jacobian to all k vectors
            Z = self._apply_jacobian(l, Q)          # [J_size, k]
 
            # QR decomposition:
            #   Q  — new orthonormal basis (renormalized directions)
            #   R  — upper triangular, diagonal holds the stretch factors
            Q, R = torch.linalg.qr(Z)               # Q: [J_size, k], R: [k, k]
 
            # Log absolute diagonal of R = log stretch for each direction
            R_diag = torch.diag(R)
            log_stretches = torch.log(torch.abs(R_diag))
            log_stretch_sum += log_stretches
 
            # Track top direction's log stretch per layer for plotting
            per_layer_log_stretches.append(log_stretches[0].item())

            self._save_layer(l, Q, R_diag, log_stretches, log_stretch_sum.clone())
 
        # Average log stretches over layers → Lyapunov exponents
        exponents = (log_stretch_sum / n_layers).tolist()
 
        # Q[:, 0] is now the converged top right singular vector of J_L @ ... @ J_1.
        # This is the input direction that experiences maximum growth through the network.
        lyapunov_vector = Q[:, 0].clone()
 
        self._save_manifest(layer_indices, per_layer_log_stretches, exponents, lyapunov_vector, k)

        self.result = LyapunovResult(
            exponents=exponents,
            per_layer_log_stretches=per_layer_log_stretches,
            lyapunov_vector=lyapunov_vector,
            layer_indices=layer_indices,
        )
 
        return self.result
 
    def alignment_with_layer_vectors(self) -> dict[int, float]:
        """
        Measures the cosine similarity between the converged Lyapunov vector
        and each layer's individual top right singular vector from power iteration.
 
        A high alignment at layer l means that layer's dominant sensitivity
        direction agrees with the global sensitivity direction of the full network.
 
        Returns:
            dict mapping layer_idx -> cosine similarity in [-1, 1]
        """
        if self.result is None:
            raise RuntimeError("Call compute() before alignment_with_layer_vectors().")
 
        global_vec = self.result.lyapunov_vector  # [J_size]
        alignments = {}
 
        for l, local_vec in self.jacobian.converged_vectors.items():
            # Both vectors are unit norm from their respective algorithms
            sim = torch.dot(
                global_vec.float(),
                local_vec.float() / local_vec.norm()
            ).abs().item()
            alignments[l] = sim
 
        return alignments
    
    @classmethod
    def load_lyapunov_run(cls, run_dir: str | Path = None) -> LyapunovResult:
        run_dir = Path(run_dir) if run_dir else cls.BASE_SAVE_DIR
        m = torch.load(run_dir / "manifest.pt", weights_only=True)
        return LyapunovResult(
            exponents=m["exponents"],
            per_layer_log_stretches=m["per_layer_log_stretches"],
            lyapunov_vector=m["lyapunov_vector"],
            layer_indices=m["layer_indices"],
        )

class LyapunovVisualizer:
    """
    Visualization tools for Lyapunov analysis of per-layer Jacobians.
 
    Provides two main plots:
      - Lyapunov exponent spectrum: bar chart of all k exponents,
        color-coded by sign (expanding vs contracting directions).
      - Layer alignment heatmap: cosine similarity between the global
        Lyapunov vector and each layer's local top singular vector,
        rendered as a 1D heatmap across layers.
 
    Usage:
        lyap = Lyapunov(jac)
        result = lyap.compute(k=8)
 
        viz = LyapunovVisualizer(lyap)
        viz.plot_spectrum()
        viz.plot_alignment_heatmap()
        viz.plot_all()           # both panels in one figure
    """
 
    # ── Palette ────────────────────────────────────────────────────────────────
    EXPAND_COLOR  = "#D85A30"   # warm coral  — positive exponent (expanding)
    CONTRACT_COLOR = "#378ADD"  # cool blue   — negative exponent (contracting)
    ZERO_LINE_COLOR = "#888780" # neutral gray — zero-crossing reference
    HEATMAP_CMAP  = "RdBu_r"   # diverging: blue → white → red, high = aligned
 
    # ── Typography & layout ────────────────────────────────────────────────────
    TITLE_SIZE   = 13
    LABEL_SIZE   = 11
    TICK_SIZE    = 10
    FIG_BG       = "#FAFAF8"
    AXES_BG      = "#FFFFFF"
    SPINE_COLOR  = "#D3D1C7"
 
    def __init__(self, lyapunov: Lyapunov):
        """
        Args:
            lyapunov: A Lyapunov instance with compute() already called.
        """
        self.lyapunov = lyapunov
        self._validate()
        self._apply_style()
 
    # ── Validation ─────────────────────────────────────────────────────────────
 
    def _validate(self) -> None:
        if self.lyapunov.result is None:
            raise RuntimeError(
                "No results found. Call lyapunov.compute() before visualizing."
            )
 
    # ── Style ──────────────────────────────────────────────────────────────────
 
    def _apply_style(self) -> None:
        """Apply a clean, publication-ready Matplotlib style."""
        plt.rcParams.update({
            "figure.facecolor":      self.FIG_BG,
            "axes.facecolor":        self.AXES_BG,
            "axes.edgecolor":        self.SPINE_COLOR,
            "axes.linewidth":        0.6,
            "axes.grid":             True,
            "axes.grid.axis":        "y",
            "grid.color":            self.SPINE_COLOR,
            "grid.linewidth":        0.4,
            "grid.linestyle":        "--",
            "grid.alpha":            0.6,
            "xtick.color":           "#5F5E5A",
            "ytick.color":           "#5F5E5A",
            "xtick.labelsize":       self.TICK_SIZE,
            "ytick.labelsize":       self.TICK_SIZE,
            "xtick.major.size":      3,
            "ytick.major.size":      3,
            "xtick.major.width":     0.5,
            "ytick.major.width":     0.5,
            "text.color":            "#2C2C2A",
            "font.family":           "sans-serif",
            "font.sans-serif":       ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
            "figure.dpi":            150,
            "savefig.dpi":           200,
            "savefig.bbox":          "tight",
            "savefig.facecolor":     self.FIG_BG,
        })
 
    def _despine(self, ax: plt.Axes, keep: tuple = ("left", "bottom")) -> None:
        """Remove unwanted spines for a cleaner look."""
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(spine in keep)
            if spine in keep:
                ax.spines[spine].set_linewidth(0.6)
                ax.spines[spine].set_color(self.SPINE_COLOR)
 
    # ── Spectrum plot ──────────────────────────────────────────────────────────
 
    def plot_spectrum(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Lyapunov exponent spectrum",
        savepath: Optional[str] = None, 
    ) -> plt.Axes:
        """
        Bar chart of the full Lyapunov exponent spectrum.
 
        Bars above zero (expanding directions) are rendered in warm coral;
        bars below zero (contracting directions) in cool blue. A dashed
        zero-line separates the two regimes.
 
        Args:
            ax:    Existing Axes to draw on. Creates a new figure if None.
            show:  Call plt.show() when done.
            title: Axes title.
 
        Returns:
            The Axes containing the plot.
        """
        result: LyapunovResult = self.lyapunov.result
        exponents = result.exponents          # list[float], length k
        k = len(exponents)
        indices = list(range(1, k + 1))       # λ₁, λ₂, …, λ_k (1-based)
 
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(max(5, k * 0.7 + 2), 4))
            fig.suptitle(title, fontsize=self.TITLE_SIZE, fontweight="normal",
                         x=0.5, y=1.01, ha="center", color="#2C2C2A")
 
        colors = [self.EXPAND_COLOR if e >= 0 else self.CONTRACT_COLOR
                  for e in exponents]
 
        bars = ax.bar(
            indices, exponents,
            color=colors,
            width=0.6,
            zorder=3,
            linewidth=0,
        )
 
        # Subtle edge on each bar for separation
        for bar, color in zip(bars, colors):
            bar.set_edgecolor(color)
            bar.set_linewidth(0.4)
 
        # Zero reference line
        ax.axhline(0, color=self.ZERO_LINE_COLOR, linewidth=0.8,
                   linestyle="--", zorder=2, alpha=0.8)
 
        # Annotate each bar with its value (skip if k is large)
        if k <= 16:
            for idx, (x, e) in enumerate(zip(indices, exponents)):
                va = "bottom" if e >= 0 else "top"
                offset = 0.008 * (max(exponents) - min(exponents) + 1e-9)
                offset = offset if e >= 0 else -offset
                ax.text(
                    x, e + offset, f"{e:.3f}",
                    ha="center", va=va,
                    fontsize=8, color="#5F5E5A",
                )
 
        # Legend patches
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=self.EXPAND_COLOR,   label="Expanding  (λ > 0)"),
            mpatches.Patch(color=self.CONTRACT_COLOR, label="Contracting (λ ≤ 0)"),
        ]
        ax.legend(handles=legend_handles, fontsize=9, frameon=False,
                  loc="upper right")
 
        ax.set_xlabel("Exponent index", fontsize=self.LABEL_SIZE, labelpad=6)
        ax.set_ylabel("λ (nats / layer)", fontsize=self.LABEL_SIZE, labelpad=6)
        ax.set_xticks(indices)
        ax.set_xticklabels([f"λ{i}" for i in indices], fontsize=self.TICK_SIZE)
        ax.tick_params(axis="x", which="both", bottom=False)
 
        self._despine(ax)
        ax.set_axisbelow(True)
 
        if standalone:
            plt.tight_layout()
            if savepath is not None:                                          
                ax.get_figure().savefig(savepath, bbox_inches="tight", dpi=200)
 
        return ax
 
    # ── Alignment heatmap ──────────────────────────────────────────────────────
 
    def plot_alignment_heatmap(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        title: str = "Layer alignment with global Lyapunov vector",
        savepath: Optional[str] = None, 
    ) -> plt.Axes:
        """
        1-D heatmap of per-layer cosine similarity with the global
        Lyapunov vector, produced by Lyapunov.alignment_with_layer_vectors().
 
        High similarity (→ 1.0) indicates that a layer's dominant local
        sensitivity direction aligns with the network-wide direction.
        The colormap is diverging (RdBu_r) centered at 0.5, so cold blues
        mark misaligned layers and warm reds mark strongly aligned ones.
 
        Args:
            ax:    Existing Axes to draw on. Creates a new figure if None.
            show:  Call plt.show() when done.
            title: Axes title.
 
        Returns:
            The Axes containing the plot.
        """
        alignments: dict[int, float] = self.lyapunov.alignment_with_layer_vectors()
        sorted_layers = sorted(alignments.keys())
        values = np.array([alignments[l] for l in sorted_layers])  # [n_layers]
 
        # Shape into a 2-row strip so the heatmap has visible height
        strip = values[np.newaxis, :]  # [1, n_layers]
 
        standalone = ax is None
        if standalone:
            n = len(sorted_layers)
            fig_w = max(8, n * 0.28 + 2)
            fig, ax = plt.subplots(figsize=(fig_w, 2.0))
            fig.suptitle(title, fontsize=self.TITLE_SIZE, fontweight="normal",
                         x=0.5, y=1.08, ha="center", color="#2C2C2A")
 
        im = ax.imshow(
            strip,
            aspect="auto",
            cmap=self.HEATMAP_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
 
        # Colorbar
        cbar = ax.get_figure().colorbar(
            im, ax=ax, orientation="vertical",
            fraction=0.015, pad=0.02,
        )
        cbar.set_label("Cosine similarity", fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8, length=2)
        cbar.outline.set_linewidth(0.4)
 
        # X-axis: layer indices (thin out if too many)
        n = len(sorted_layers)
        step = max(1, n // 20)
        shown_ticks = list(range(0, n, step))
        ax.set_xticks(shown_ticks)
        ax.set_xticklabels([str(sorted_layers[i]) for i in shown_ticks],
                           fontsize=self.TICK_SIZE)
        ax.set_xlabel("Layer index", fontsize=self.LABEL_SIZE, labelpad=6)
 
        # Y-axis: hide — single row
        ax.set_yticks([])
 
        # Annotate each cell with its value if compact enough
        if n <= 24:
            for j, v in enumerate(values):
                text_color = "white" if (v < 0.3 or v > 0.75) else "#2C2C2A"
                ax.text(j, 0, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=text_color, fontweight="500")
 
        self._despine(ax, keep=())  # no spines on heatmap
        ax.tick_params(axis="both", which="both",
                       bottom=False, left=False, top=False, right=False)
 
        if standalone:
            plt.tight_layout()
            if savepath is not None:                                          
                ax.get_figure().savefig(savepath, bbox_inches="tight", dpi=200)
 
        return ax
 
    # ── Combined figure ────────────────────────────────────────────────────────
 
    def plot_all(
        self,
        show: bool = True,
        figsize: Optional[tuple] = None,
        suptitle: str = "Lyapunov analysis",
    ) -> plt.Figure:
        """
        Render both plots in a single figure:
          - Top panel:    Lyapunov exponent spectrum
          - Bottom panel: Layer alignment heatmap
 
        Args:
            show:     Call plt.show() when done.
            figsize:  Override the automatic figure size.
            suptitle: Figure-level title.
 
        Returns:
            The Figure object.
        """
        result: LyapunovResult = self.lyapunov.result
        k = len(result.exponents)
        n_layers = len(result.layer_indices)
        fig_w = figsize[0] if figsize else max(10, max(k, n_layers) * 0.35 + 3)
        fig_h = figsize[1] if figsize else 7.5
 
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=self.FIG_BG)
        fig.suptitle(suptitle, fontsize=self.TITLE_SIZE + 1, fontweight="normal",
                     y=1.01, color="#2C2C2A")
 
        gs = GridSpec(
            2, 1,
            figure=fig,
            height_ratios=[3, 1],
            hspace=0.55,
        )
 
        ax_spectrum = fig.add_subplot(gs[0])
        ax_heatmap  = fig.add_subplot(gs[1])
 
        self.plot_spectrum(
            ax=ax_spectrum,
            show=False,
            title="Lyapunov exponent spectrum",
        )
        ax_spectrum.set_title(
            "Lyapunov exponent spectrum",
            fontsize=self.TITLE_SIZE, pad=8, color="#2C2C2A",
        )
 
        self.plot_alignment_heatmap(
            ax=ax_heatmap,
            show=False,
            title="Layer alignment with global Lyapunov vector",
        )
        ax_heatmap.set_title(
            "Layer alignment with global Lyapunov vector",
            fontsize=self.TITLE_SIZE, pad=8, color="#2C2C2A",
        )
 
        if show:
            plt.show()
 
        return fig
 
    # ── Export ─────────────────────────────────────────────────────────────────
 
    def save(self, path: str, **savefig_kwargs) -> None:
        """
        Save the combined figure to disk.
 
        Args:
            path: Output file path (e.g. "lyapunov.png" or "lyapunov.pdf").
            **savefig_kwargs: Forwarded to plt.savefig().
        """
        fig = self.plot_all(show=False)
        savefig_kwargs.setdefault("bbox_inches", "tight")
        savefig_kwargs.setdefault("dpi", 200)
        fig.savefig(path, **savefig_kwargs)
        plt.close(fig)
        print(f"Saved to {path}")