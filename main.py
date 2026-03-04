import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import create_causal_mask
import dataclasses
from dataclasses import dataclass
import contextlib
from tqdm import tqdm
from pprint import pprint

@dataclass
class LayerInputs:
    hidden_states: torch.Tensor
    causal_mask: torch.Tensor
    position_ids: torch.Tensor
    position_embeddings: tuple
    cache_position: torch.Tensor


class MyModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            output_hidden_states=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(self.model)
        self.model.gradient_checkpointing_disable()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.lm_head = self.model.lm_head
        self.norm = self.model.model.norm 
        

    def tokenize(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        return tokens["input_ids"]
    
    def prepare_layer_inputs(self, input_ids: torch.Tensor) -> LayerInputs:
        h = self.model.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=input_ids.device)

        causal_mask = create_causal_mask(
            config=self.model.config,
            input_embeds=h,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = self.model.model.rotary_emb(h, position_ids=position_ids)

        return LayerInputs(
            hidden_states=h,
            causal_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
    
    def forward_layer(self, layer_idx: int, inputs: LayerInputs, no_grad: bool = True) -> LayerInputs:
        layer = self.model.model.layers[layer_idx]
        context = torch.no_grad() if no_grad else contextlib.nullcontext()
        with context:
            h = layer(
                hidden_states=inputs.hidden_states,
                attention_mask=inputs.causal_mask,
                position_embeddings=inputs.position_embeddings,
                position_ids=inputs.position_ids,
                past_key_values=None,
                cache_position=inputs.cache_position,
            )

        return dataclasses.replace(inputs, hidden_states=h)

    def _probe_forward(self, inputs, hl):
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
            logits = self.lm_head(self.norm(outputs.hidden_states[hl]))
            probs = torch.softmax(logits, dim=-1)

        return probs
    
      
        

    def probe_layer(self, inputs, hl=-1, token_id=None):
        probs = self._probe_forward(inputs, hl)          # [1, seq_len, vocab]
        next_token_probs = probs[:, -1, :]         # [1, vocab]

        # argmax info
        argmax_id = torch.argmax(next_token_probs, dim=-1).item()
        argmax_prob = next_token_probs[0, argmax_id].item()

        # specific token prob (optional)
        token_prob = None
        if token_id is not None:
            token_prob = next_token_probs[0, token_id].item()

        return token_prob, argmax_id, argmax_prob

            

class KFAC:
    def __init__(self, mmodel, layers, target_token_id):
        self.mmodel = mmodel
        self.layers = layers
        self.target_token_id = target_token_id
        
        self.factors = {}
        self.eigenvalues = {}
        self.max_eigenvalues = {}
        self.top_eigenvectors = {}
        self.gradient_projections = {}

    # --- Hook setup ---
    @staticmethod
    def register_kfac_hooks(layer):
        """Register hooks on all nn.Linear modules in a layer. Returns handles and storage dicts."""
        activations = {}
        gradients = {}
        handles = []

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # print(name, ":", module)
                def fwd_hook(mod, inp, out, _name=name):
                    # inp[0] is (batch, seq, d_in)
                    activations[_name] = inp[0].detach()

                def bwd_hook(mod, grad_inp, grad_out, _name=name):
                    # grad_out[0] is dL/d(output), (batch, seq, d_out)
                    gradients[_name] = grad_out[0].detach()

                handles.append(module.register_forward_hook(fwd_hook))
                handles.append(module.register_full_backward_hook(bwd_hook))

        return handles, activations, gradients
    
    def compute_layer_gradients(
        mmodel,
        layer_inputs: LayerInputs,
        layer_idx: int,
        target_token_id: int,
        layers: list,
    ) -> tuple[dict, dict]:
        """
        Runs a forward pass from layer_idx to the final output, computes
        the log probability of target_token_id, and backpropagates.
        Returns (activations, gradients) dicts keyed by linear layer name.
        """
        mmodel.model.zero_grad()

        layer = layers[layer_idx]
        handles, activations, gradients = KFAC.register_kfac_hooks(layer)

        # Detach input and make it a fresh leaf so gradients don't flow further back
        h_in = layer_inputs.hidden_states.detach().requires_grad_(True)
        inputs_for_layer = dataclasses.replace(layer_inputs, hidden_states=h_in)

        # Forward through layer l
        out = mmodel.forward_layer(layer_idx, inputs_for_layer, no_grad=False)
        h = out.hidden_states

        # Forward through remaining layers
        for i in range(layer_idx + 1, len(layers)):
            out_i = mmodel.forward_layer(i, dataclasses.replace(layer_inputs, hidden_states=h), no_grad=False)
            h = out_i.hidden_states

        # Compute loss and backprop
        logits = mmodel.lm_head(mmodel.norm(h))
        log_prob = torch.log_softmax(logits[0, -1, :], dim=-1)
        score = log_prob[target_token_id]
        score.backward()

        # Cleanup
        for handle in handles:
            handle.remove()

        return activations, gradients
    
    
    @staticmethod
    def compute_kfac_factors(activations: dict, gradients: dict) -> dict:
        """
        Computes KFAC factors A and G for each linear layer.
        A = a_flat.T @ a_flat  (d_in, d_in)
        G = g_flat.T @ g_flat  (d_out, d_out)
        Returns a dict keyed by layer name with (A, G) tuples.
        """
        factors = {}

        for name in activations:
            a = activations[name]
            g = gradients[name]

            a_flat = a.reshape(-1, a.shape[-1])
            g_flat = g.reshape(-1, g.shape[-1])
            print(a_flat.abs().max(), a_flat.abs().mean())
            
            A = (a_flat.double().T @ a_flat.double())
            G = (g_flat.double().T @ g_flat.double())

            weight_grad = (g_flat.T @ a_flat).view(-1)

            factors[name] = (A, G, weight_grad)

        return factors
    
    
    def collect_factors(self, layer_inputs: LayerInputs, verify: bool = False):
        """
        Runs forward/backward passes across all layers to collect KFAC factors.
        Populates self.factors as {layer_idx: {linear_name: (A, G)}}.
        """
        for l, layer in enumerate(tqdm(self.layers)):
            activations, gradients = KFAC.compute_layer_gradients(
                self.mmodel, layer_inputs, l, self.target_token_id, self.layers
            )

            if verify:
                results = KFAC.verify_reconstruction(layer, activations, gradients)
                for name, diff in results.items():
                    if diff is not None:
                        print(f"  {name:40s}  max_diff: {diff[0]:.6f}  mean_diff: {diff[1]:.6f}")
                    else:
                        print(f"  {name:40s}  no .grad found")

            print(l, ": ")
            factors = KFAC.compute_kfac_factors(activations, gradients)

            # Compute and store only the small derived quantities, discard raw factors
            self.eigenvalues[l] = {}
            self.max_eigenvalues[l] = {}
            self.top_eigenvectors[l] = {}
            self.gradient_projections[l] = {}

            for name, (A, G, weight_grad) in factors.items():
                eigs_A, vecs_A = torch.linalg.eigh(A)
                eigs_G, vecs_G = torch.linalg.eigh(G)

                eig_A = eigs_A[-1].item()
                eig_G = eigs_G[-1].item()
                top_vec_A = vecs_A[:, -1]
                top_vec_G = vecs_G[:, -1]
                top_vec = torch.outer(top_vec_G, top_vec_A).view(-1)

                self.eigenvalues[l][name] = (eig_A, eig_G)
                self.max_eigenvalues[l][name] = eig_A * eig_G
                self.top_eigenvectors[l][name] = top_vec
                self.gradient_projections[l][name] = torch.dot(
                    weight_grad.double(), top_vec
                ).item()

            layer_inputs = self.mmodel.forward_layer(l, layer_inputs)

    

    @staticmethod
    def verify_reconstruction(layer: nn.Module, activations: dict, gradients: dict) -> dict:
        """
        For each linear layer, checks whether g.T @ a matches module.weight.grad.
        Returns a dict keyed by layer name with (max_diff, mean_diff) tuples.
        """
        results = {}

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                a = activations[name]
                g = gradients[name]

                a_flat = a.reshape(-1, a.shape[-1])
                g_flat = g.reshape(-1, g.shape[-1])

                reconstructed = g_flat.T @ a_flat
                actual = module.weight.grad

                if actual is not None:
                    diff = (reconstructed.to(actual.device) - actual).abs()
                    results[name] = (diff.max().item(), diff.mean().item())
                else:
                    results[name] = None

        return results
    
    def compute_eigenvalues(self):
        """
        Computes the top eigenvalue of A and G for each linear layer in each transformer layer.
        Populates self.eigenvalues as {layer_idx: {linear_name: (eig_A, eig_G)}}
        and self.max_eigenvalues as {layer_idx: {linear_name: eig_A * eig_G}}.
        """
        for l, layer_factors in tqdm(self.factors.items()):
            self.eigenvalues[l] = {}
            self.max_eigenvalues[l] = {}
            self.top_eigenvectors[l] = {}

            for name, (A, G, _) in layer_factors.items():
        
                # eigvalsh returns eigenvalues in ascending order, so take the last one
                # eig_A = torch.linalg.eigvalsh(A)[-1].item()
                # eig_G = torch.linalg.eigvalsh(G)[-1].item()
                eigs_A, vecs_A = torch.linalg.eigh(A)
                eigs_G, vecs_G = torch.linalg.eigh(G)

                # Max eigenvalues of A and G
                eig_A = eigs_A[-1].item()
                eig_G = eigs_G[-1].item()

                # Max eigenvectors of A and G
                top_vec_A = vecs_A[:, -1]   # (d_in,)
                top_vec_G = vecs_G[:, -1]   # (d_out,)

                self.eigenvalues[l][name] = (eig_A, eig_G)
                self.max_eigenvalues[l][name] = eig_A * eig_G
                self.top_eigenvectors[l][name] = torch.outer(top_vec_G, top_vec_A).view(-1)

    def compute_gradient_projections(self):
        for l, layer_factors in self.factors.items():
            self.gradient_projections[l] = {}
            for name, (A, G, weight_grad) in layer_factors.items():
                top_eigvec = self.top_eigenvectors[l][name]
                self.gradient_projections[l][name] = torch.dot(
                    weight_grad.double(),
                    top_eigvec
                ).item()


    def run(self, layer_inputs: LayerInputs, verify: bool = False):
        self.collect_factors(layer_inputs, verify)
        # self.compute_eigenvalues()
        # self.compute_gradient_projections()
        return self.max_eigenvalues, self.gradient_projections
    




class KFACVisualizer:
    # Canonical projection order: attention first, then MLP
    PROJ_ORDER = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    PROJ_LABELS = ["Q", "K", "V", "O", "Gate", "Up", "Down"]

    _CURV_COLORS = [
        "#0a0e1a",   # near-black navy  (low curvature / flat)
        "#1a2744",
        "#2d3f6b",
        "#4b3f8c",   # indigo
        "#8b3a9e",
        "#c94f4f",   # warm red
        "#e8882a",   # amber
        "#f5d060",   # gold
        "#ffffff",   # white-hot        (high curvature / sharp)
    ]

    CURVATURE_CMAP = LinearSegmentedColormap.from_list("curvature", _CURV_COLORS, N=512)

    
    ARROW_TOWARD = "#00e5ff"   
    ARROW_AWAY   = "#ff3cac"   


    def __init__(self, kfac: KFAC):
        self.kfac = kfac

    def _build_matrix(self, scale_by_curvature: bool = False) -> np.ndarray:
        """
        Returns a (n_layers, n_proj) matrix of gradient projections,
        optionally scaled by max_eigenvalue.
        """
        n_layers = len(self.kfac.gradient_projections)
        n_proj = len(self.PROJ_ORDER)
        mat = np.zeros((n_layers, n_proj))

        for l in range(n_layers):
            for j, name in enumerate(self.PROJ_ORDER):
                proj = self.kfac.gradient_projections.get(l, {}).get(name)
                if proj is None:
                    continue
                if scale_by_curvature:
                    eig = self.kfac.max_eigenvalues.get(l, {}).get(name, 1.0)
                    mat[l, j] = proj * eig

                else:
                    mat[l, j] = proj

        if scale_by_curvature:
            col_maxes = np.abs(mat).max(axis=0, keepdims=True)
            col_maxes[col_maxes == 0] = 1.0
            mat = mat / col_maxes

        return mat

    def plot_heatmap(self, scale_by_curvature: bool = False, ax=None):
        """
        Heatmap: rows = layers, cols = projections.
        Color = gradient projection (diverging, centered at zero).
        Positive = gradient pushing toward target label.
        Negative = gradient pushing away.
        """
        mat = self._build_matrix(scale_by_curvature)
        vmax = np.abs(mat).max()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 12))
        else:
            fig = ax.get_figure()

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_xticks(range(len(self.PROJ_LABELS)))
        ax.set_xticklabels(self.PROJ_LABELS, fontsize=11)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([f"L{l}" for l in range(mat.shape[0])], fontsize=8)
        ax.set_xlabel("Projection", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)

        title = "Gradient Projection onto Top FIM Eigenvector"
        if scale_by_curvature:
            title += "\n(scaled by max eigenvalue)"
        ax.set_title(title, fontsize=13, pad=12)

        # Divider between attention and MLP columns
        ax.axvline(x=3.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
        ax.text(1.5, -1.2, "Attention", ha="center", fontsize=10, color="gray")
        ax.text(5.0, -1.2, "MLP", ha="center", fontsize=10, color="gray")

        plt.colorbar(im, ax=ax, shrink=0.6, label="← away from label  |  toward label →")
        return fig, ax

    def plot_layer_summary(self, ax=None):
        """
        Bar chart: per-layer sum of |gradient_projection| split into
        attention vs MLP, showing which layers are most active.
        """
        n_layers = len(self.kfac.gradient_projections)
        attn_names = self.PROJ_ORDER[:4]
        mlp_names  = self.PROJ_ORDER[4:]

        attn_sums = np.zeros(n_layers)
        mlp_sums  = np.zeros(n_layers)

        for l in range(n_layers):
            layer_data = self.kfac.gradient_projections.get(l, {})
            attn_sums[l] = sum(abs(layer_data.get(n, 0)) for n in attn_names)
            mlp_sums[l]  = sum(abs(layer_data.get(n, 0)) for n in mlp_names)

        layers = np.arange(n_layers)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.get_figure()

        ax.bar(layers, attn_sums, label="Attention", color="#4C72B0", alpha=0.85)
        ax.bar(layers, mlp_sums,  label="MLP",       color="#DD8452", alpha=0.85,
               bottom=attn_sums)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("|Gradient Projection| (summed)", fontsize=12)
        ax.set_title("Per-Layer Classification Sensitivity", fontsize=13)
        ax.set_xticks(layers)
        ax.set_xticklabels([str(l) for l in layers], fontsize=7)
        ax.legend()
        return fig, ax
    
    def plot_quiver(
        self,
        save_path: str = None,
        figsize: tuple = (13, 11),
        arrow_scale: float = 0.38,   # fraction of a cell that the longest arrow occupies
    ):
        """
        Quiver plot: curvature heatmap background + directional gradient arrows.

        Parameters
        ----------
        save_path    : optional file path to save the figure (PNG / PDF).
        figsize      : matplotlib figure size.
        arrow_scale  : controls max arrow length relative to cell width (0–1).
        """
        PROJ_ORDER  = self.PROJ_ORDER   # list of 7 projection name strings
        PROJ_LABELS = self.PROJ_LABELS  # ["Q","K","V","O","Gate","Up","Down"]

        n_layers = len(self.kfac.gradient_projections)
        n_proj   = len(PROJ_ORDER)

        # ------------------------------------------------------------------
        # 1. Build raw matrices
        # ------------------------------------------------------------------
        eig_mat  = np.zeros((n_layers, n_proj))   # max eigenvalue (curvature)
        proj_mat = np.zeros((n_layers, n_proj))   # signed gradient projection

        for l in range(n_layers):
            for j, name in enumerate(PROJ_ORDER):
                gp  = self.kfac.gradient_projections.get(l, {}).get(name)
                eig = self.kfac.max_eigenvalues.get(l, {}).get(name)
                if gp  is not None: proj_mat[l, j] = gp
                if eig is not None: eig_mat[l, j]  = eig

        # ------------------------------------------------------------------
        # 2. Normalise curvature: column-wise so each projection type uses
        #    the full colour range (prevents down_proj's huge eigenvalue from
        #    saturating everything else).
        # ------------------------------------------------------------------
        eig_display = eig_mat.copy()
        col_max = eig_display.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1.0
        eig_display = eig_display / col_max          # each column ∈ [0, 1]

        # ------------------------------------------------------------------
        # 3. Normalise arrow lengths: row-wise (per layer) so every layer
        #    has at least one full-length arrow — readable even for quiet layers.
        # ------------------------------------------------------------------
        abs_proj = np.abs(proj_mat)
        row_max  = abs_proj.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        arrow_len_norm = abs_proj / row_max          # each row ∈ [0, 1]

        # ------------------------------------------------------------------
        # 4. Build quiver components
        #    U = signed normalised length (positive = right = toward label)
        #    V = 0 (purely horizontal arrows)
        # ------------------------------------------------------------------
        signs = np.sign(proj_mat)
        signs[signs == 0] = 1.0
        U = signs * arrow_len_norm   # ∈ [-1, 1]
        V = np.zeros_like(U)

        # Grid positions (cell centres)
        X, Y = np.meshgrid(np.arange(n_proj), np.arange(n_layers))

        # ------------------------------------------------------------------
        # 5. Figure
        # ------------------------------------------------------------------
        plt.rcParams.update({
            "font.family": "monospace",
            "text.color":  "#e8e8e8",
            "axes.labelcolor": "#e8e8e8",
            "xtick.color": "#e8e8e8",
            "ytick.color": "#e8e8e8",
        })

        fig, ax = plt.subplots(figsize=figsize, facecolor="#080c14")
        ax.set_facecolor("#080c14")

        # ------------------------------------------------------------------
        # 6. Background heatmap (imshow, extent centred on integer grid pts)
        # ------------------------------------------------------------------
        im = ax.imshow(
            eig_display,
            aspect="auto",
            cmap=self.CURVATURE_CMAP,
            vmin=0, vmax=1,
            extent=[-0.5, n_proj - 0.5, n_layers - 0.5, -0.5],
            interpolation="bilinear",
            alpha=0.92,
            zorder=0,
        )

        # ------------------------------------------------------------------
        # 7. Quiver — split toward/away so each gets its own colour
        # ------------------------------------------------------------------
        cell_w = 1.0   # one unit = one cell in data coords
        # scale: when |U|=1, arrow spans arrow_scale * cell_w
        q_scale = 1.0 / arrow_scale   # matplotlib quiver scale param (inverse)

        toward_mask = (U >= 0)
        away_mask   = (U <  0)

        def _quiver(mask, color):
            if not mask.any():
                return
            ax.quiver(
                X[mask], Y[mask],
                U[mask], V[mask],
                color=color,
                scale=q_scale,
                scale_units="x",
                width=0.006,
                headwidth=4.5,
                headlength=5.0,
                headaxislength=4.2,
                alpha=0.95,
                zorder=3,
                pivot="mid",
            )

        _quiver(toward_mask, self.ARROW_TOWARD)
        _quiver(away_mask,   self.ARROW_AWAY)

        # ------------------------------------------------------------------
        # 8. Grid lines to delineate cells
        # ------------------------------------------------------------------
        for x in np.arange(-0.5, n_proj, 1):
            ax.axvline(x, color="#ffffff", lw=0.25, alpha=0.18, zorder=1)
        for y in np.arange(-0.5, n_layers, 1):
            ax.axhline(y, color="#ffffff", lw=0.25, alpha=0.18, zorder=1)

        # Thicker divider between attention (cols 0-3) and MLP (cols 4-6)
        ax.axvline(3.5, color="#ffffff", lw=1.4, alpha=0.55, linestyle="--", zorder=2)

        # ------------------------------------------------------------------
        # 9. Axes, ticks, labels
        # ------------------------------------------------------------------
        ax.set_xlim(-0.5, n_proj - 0.5)
        ax.set_ylim(n_layers - 0.5, -0.5)   # layer 0 at top

        ax.set_xticks(range(n_proj))
        ax.set_xticklabels(PROJ_LABELS, fontsize=11, fontfamily="monospace")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l:02d}" for l in range(n_layers)], fontsize=7.5,
                        fontfamily="monospace")

        ax.set_xlabel("Projection", fontsize=12, labelpad=8)
        ax.set_ylabel("Layer", fontsize=12, labelpad=8)

        # Section labels above x-axis
        ax.text(1.5,  -1.05, "── ATTENTION ──", ha="center", fontsize=9,
                color="#aac8e8", fontfamily="monospace", transform=ax.transData,
                clip_on=False)
        ax.text(5.0,  -1.05, "──── MLP ────", ha="center", fontsize=9,
                color="#e8c8aa", fontfamily="monospace", transform=ax.transData,
                clip_on=False)

        # ------------------------------------------------------------------
        # 10. Title
        # ------------------------------------------------------------------
        title_pe = [patheffects.withStroke(linewidth=3, foreground="#080c14")]
        ax.set_title(
            "KFAC Loss Landscape  ·  Gradient Flow toward Classification Boundary",
            fontsize=12, pad=18, color="#e8e8e8",
            fontfamily="monospace",
            path_effects=title_pe,
        )

        # ------------------------------------------------------------------
        # 11. Colourbars
        # ------------------------------------------------------------------
        # Curvature colourbar (right)
        cbar_curv = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, aspect=28)
        cbar_curv.set_label("Curvature  (max eigenvalue, col-normalised)",
                            fontsize=9, color="#e8e8e8", fontfamily="monospace")
        cbar_curv.ax.yaxis.set_tick_params(color="#e8e8e8")
        plt.setp(cbar_curv.ax.yaxis.get_ticklabels(), color="#e8e8e8",
                fontfamily="monospace", fontsize=8)
        cbar_curv.outline.set_edgecolor("#444")

        # Arrow legend (manual)
        from matplotlib.patches import FancyArrow
        legend_y = 1.045
        ax.annotate("", xy=(0.30, legend_y), xytext=(0.22, legend_y),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=self.ARROW_TOWARD, lw=1.8))
        ax.text(0.31, legend_y, "toward label", transform=ax.transAxes,
                color=self.ARROW_TOWARD, fontsize=9, va="center", fontfamily="monospace")

        ax.annotate("", xy=(0.56, legend_y), xytext=(0.64, legend_y),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=self.ARROW_AWAY, lw=1.8))
        ax.text(0.65, legend_y, "away from label", transform=ax.transAxes,
                color=self.ARROW_AWAY, fontsize=9, va="center", fontfamily="monospace")

        # ------------------------------------------------------------------
        # 12. Subtle note about row-normalisation
        # ------------------------------------------------------------------
        fig.text(0.5, 0.01,
                "Arrow length: |gradient projection|, row-normalised per layer  "
                "·  Background: loss-landscape curvature",
                ha="center", fontsize=8, color="#666e80", fontfamily="monospace")

        plt.tight_layout(rect=[0, 0.025, 1, 0.97])

        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved → {save_path}")

        plt.show()
        return fig, ax

    def plot_all(self, scale_by_curvature: bool = False, save_path: str = None):
        """Renders heatmap + layer summary in one figure."""
        fig = plt.figure(figsize=(14, 16))
        gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)

        ax_heat = fig.add_subplot(gs[0])
        ax_bar  = fig.add_subplot(gs[1])

        self.plot_heatmap(scale_by_curvature=scale_by_curvature, ax=ax_heat)
        self.plot_layer_summary(ax=ax_bar)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")

        plt.show()
        return fig



if __name__ == '__main__':
    prompt = "'This movie was terrible.'\n\n " \
    "Was the previous review positive or negative? " \
    "The previous review was "

    mmodel = MyModel()
    
    
    # Necessary to mimic a normal forward pass
    mmodel.model.config._attn_implementation = "eager"

    
    input_ids = mmodel.tokenize(prompt)
    seq_len = input_ids.shape[1]
    negative_ids = mmodel.tokenize(" negative").to(mmodel.model.device)
    # tokens = mmodel.tokenizer.encode(" negative", add_special_tokens=False)
    target_token_id = mmodel.tokenizer.encode(" negative", add_special_tokens=False)


    jacobians = []
    layers = mmodel.model.model.layers

    layer_inputs = mmodel.prepare_layer_inputs(input_ids)
    J_size = seq_len * mmodel.model.config.hidden_size
    chunk_size = 256


    kfac = KFAC(mmodel, layers, target_token_id)
    max_eigenvalues, gradient_projections = kfac.run(layer_inputs)

    viz = KFACVisualizer(kfac)

    # Raw projection heatmap + layer summary
    viz.plot_all(save_path="kfac_raw.png")

    # Curvature-weighted version
    viz.plot_all(scale_by_curvature=True, save_path="kfac_weighted.png")
    
    pprint(max_eigenvalues)

    print("Gradient Projections")
    pprint(gradient_projections)

    # for l, layer in enumerate(tqdm(layers)):
    #     print(f"\n=== Layer {l} ===")
    #     os.makedirs(f"jacobians/layer_{l}", exist_ok=True)
        
    #     mmodel.model.zero_grad()
    #     #FIM    
    #     handles, activations, gradients = KFAC.register_kfac_hooks(layer)
    #     # Forward through target layer WITH grad
    #     h_in = layer_inputs.hidden_states.detach().requires_grad_(True)
    #     inputs_for_layer = dataclasses.replace(layer_inputs, hidden_states=h_in)
    #     target_out = mmodel.forward_layer(l, inputs_for_layer, no_grad=False)

    #     h = target_out.hidden_states
        
    #     # Forward pass through rest of model (layer+1)
    #     for i in range(l + 1, len(layers)):
    #         layer_inputs_i = mmodel.forward_layer(i, dataclasses.replace(layer_inputs, hidden_states=h), no_grad=False)
    #         h = layer_inputs_i.hidden_states

    #     logits = mmodel.lm_head(mmodel.norm(h))  # (batch, seq, vocab)

    #     # 4. Compute a loss to backprop (Fisher uses sampled labels)
    #     log_prob = torch.log_softmax(logits[0, -1, :], dim=-1).to(mmodel.model.device)
    #     score = log_prob[negative_ids[0][1]]
    #     print(f"this module's score is: {score}")

    #     # 5. Backward — hooks fire and capture g_out for each Linear
    #     score.backward()
        

    #     # 6. Inspect what we captured
    #     for name in activations:
    #         a = activations[name]  # (batch, seq, d_in)
    #         g = gradients[name]    # (batch, seq, d_out)
    #         print(f"  {name:40s}  a: {a.shape}  g: {g.shape}")

    #     #### KFAC test
    #     for name, module in layer.named_modules():
    #         if isinstance(module, nn.Linear) and name in activations:
    #             a = activations[name]  # (1, T, d_in)
    #             g = gradients[name]    # (1, T, d_out)
                
    #             # Flatten batch and seq dims
    #             a_flat = a.reshape(-1, a.shape[-1])   # (T, d_in)
    #             g_flat = g.reshape(-1, g.shape[-1])   # (T, d_out)
                
    #             # KFAC-reconstructed weight grad
    #             reconstructed = g_flat.T @ a_flat     # (d_out, d_in)
                
    #             # PyTorch's actual weight grad
    #             actual = module.weight.grad            # (d_out, d_in)
                
    #             if actual is not None:
    #                 diff = (reconstructed.to(actual.device) - actual).abs()
    #                 print(f"{name:40s}  max_diff: {diff.max().item():.6f}  mean_diff: {diff.mean().item():.6f}")
    #             else:
    #                 print(f"{name:40s}  no .grad found (weight may not require grad)")

    #     # 7. Cleanup hooks
    #     for handle in handles:
    #         handle.remove()

    #     # 8. Advance layer_inputs for the next iteration (no grad, as before)
    #     layer_inputs = mmodel.forward_layer(l, layer_inputs)




        #JACOBIAN
        # for chunk_idx, output_start in enumerate(range(0, J_size, chunk_size)):

        #     output_end = min(output_start + chunk_size, J_size)
        #     h_in = layer_inputs.hidden_states.detach().requires_grad_(True)

        #     inputs_for_jacobian = dataclasses.replace(layer_inputs, hidden_states=h_in)

        #     def forward_partial(h):
        #         return mmodel.forward_layer(l, dataclasses.replace(inputs_for_jacobian, hidden_states=h), no_grad=False).hidden_states[0].reshape(-1)[output_start:output_end]

        #     J = torch.autograd.functional.jacobian(forward_partial, h_in)
        #     J = J.reshape(output_end - output_start, J_size)
            
        #     J_path = f"jacobians/layer_{l}/shard_{chunk_idx}.pt"
        #     # torch.save(J.float().cpu(), J_path)
            
        #     print(J.shape)
        #     del J
        #     torch.cuda.empty_cache()
            
        
        # jacobians.append(J)

        # layer_inputs = mmodel.forward_layer(l, layer_inputs)  # advance with no_grad
        # Advance h to the next layer's output (no grad needed)
        # with torch.no_grad():
           
        #     h = layer(
        #         h_in,
        #         attention_mask=causal_mask,
        #         position_embeddings=position_embeddings,
        #         cache_position=cache_position,
        #     )

        # Probe: project the last token's hidden state through norm + lm_head
        # with torch.no_grad():
        #     last_token_h = layer_inputs.hidden_states[:, -1, :]           # shape: [1, hidden_size]
        #     normed = mmodel.norm(last_token_h)
        #     logits = mmodel.lm_head(normed)       # shape: [1, vocab_size]
        #     probs = torch.softmax(logits, dim=-1)

        #     top_k = 10
        #     top_probs, top_ids = torch.topk(probs[0], top_k)
        #     argmax_id = top_ids[0].item()
        #     argmax_token = mmodel.tokenizer.decode([argmax_id])
        #     print(f"\n--- After layer {l} ---")
        #     print(f"Argmax token: '{argmax_token}' (id={argmax_id}, prob={top_probs[0].item():.4f})")

        

            
        # word_prob = 1.0
        # for idx, token_id in enumerate(negative_ids[0,1:]):
        #     token_prob, argmax_id, argmax_prob = mmodel.probe_layer(input_ids, hl=l, token_id=token_id.item())

        #     word_prob *= token_prob
        #     # inputs = torch.cat([inputs, torch.tensor([[token_id]], device=inputs.device)], dim=1)

        #     # Print current token and its probability
        #     current_token = mmodel.tokenizer.decode(token_id)
        #     argmax_token = mmodel.tokenizer.decode(argmax_id)
        #     print("===Original====")
        #     print(
        #         # f"Target token: '{current_token}' | "
        #         f"P(target)={token_prob} | "
        #         f"Argmax: '{argmax_token}' | "
        #         f"P(argmax)={argmax_prob}"
        #     )
        
        # print(f"Word probability up to this layer: {word_prob}")


    _, argmax_id, argmax_prob = mmodel.probe_layer(input_ids)
    
    
    final_token = mmodel.tokenizer.decode(argmax_id)


    print(f"\n=== Output (lm_head) ===")

    print(f"Most likely token: '{final_token}' | Probability: {argmax_prob}")
 
   
     