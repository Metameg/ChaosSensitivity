import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import create_causal_mask
import dataclasses
from dataclasses import dataclass
import contextlib
from tqdm import tqdm
from pprint import pprint
from KFAC import KFAC, KFACVisualizer
from data_types import LayerInputs
from Jacobian import Jacobian, JacobianVisualizer


class MyModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            output_hidden_states=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.model.gradient_checkpointing_disable()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.lm_head = self.model.lm_head
        self.norm = self.model.model.norm 

        # Necessary to mimic a normal forward pass
        self.model.config._attn_implementation = "eager"

        

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

            




if __name__ == '__main__':
    prompt = "'This movie was terrible.'\n\n " \
    "Was the previous review positive or negative? " \
    "The previous review was "

    mmodel = MyModel()
    
    
    
    
    input_ids = mmodel.tokenize(prompt)
    seq_len = input_ids.shape[1]
    negative_ids = mmodel.tokenize(" negative").to(mmodel.model.device)
    # tokens = mmodel.tokenizer.encode(" negative", add_special_tokens=False)
    target_token_id = mmodel.tokenizer.encode(" negative", add_special_tokens=False)



    layer_inputs = mmodel.prepare_layer_inputs(input_ids)
    J_size = seq_len * mmodel.model.config.hidden_size
    


    # layers = mmodel.model.model.layers
    # kfac = KFAC(mmodel, layers, target_token_id)
    # max_eigenvalues, gradient_projections = kfac.run(layer_inputs)

    # kfac_viz = KFACVisualizer(kfac)

    # # Raw projection heatmap + layer summary
    # kfac_viz.plot_all(save_path="kfac_raw.png")

    # # Curvature-weighted version
    # # kfac_viz.plot_all(scale_by_curvature=True, save_path="kfac_weighted.png")
    # kfac_viz.plot_layer_summary()
    # kfac_viz.plot_quiver(save_path="quiver.png")

    jac = Jacobian(mmodel, 
                   layer_inputs,
                   save_dir="jacobians_threaded",
                   chunk_size=256,
                   start_layer=1)
    jac.compute()
    # Reload shards and rerun power iteration for all layers
    # n_layers = len(mmodel.model.model.layers)
    # for l in range(0, 3):
    #     jac._reload_manifest_from_disk(l)
    #     jac.spectral_norm_from_disk(l)  # populates spectral_norms and converged_vectors


    # # Decode tokens for the sensitivity heatmap x-axis
    # token_ids = input_ids[0].tolist()
    # token_labels = [mmodel.tokenizer.decode([t]) for t in token_ids]

    # # jac_viz = JacobianVisualizer(jac)
    # # jac_viz.plot_spectral_profile(kfac=kfac, save_path="plots/spectral_profile_jacobian.png")
    # # jac_viz.plot_sensitivity_heatmap(token_labels=token_labels, save_path="plots/jacobian_sensitivity_heatmap.png")
    # # jac_viz.plot_correlation_scatter(kfac, save_path="plots/eigen-singular-correlation.png")
    


  

        # layer_inputs = mmodel.forward_layer(l, layer_inputs)  # advance with no_grad
        # # Advance h to the next layer's output (no grad needed)
        # with torch.no_grad():
           
        #     h = layer(
        #         h_in,
        #         attention_mask=layer_inputs.causal_mask,
        #         position_embeddings=layer_inputs.position_embeddings,
        #         cache_position=layer_inputs.cache_position,
        #     )

        # # Probe: project the last token's hidden state through norm + lm_head
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
 
   
     