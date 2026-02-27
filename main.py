import os
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

            factors[name] = (A, G)

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
            self.factors[l] = KFAC.compute_kfac_factors(activations, gradients)

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

            for name, (A, G) in layer_factors.items():
        
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


    def run(self, layer_inputs: LayerInputs, verify: bool = False):
        self.collect_factors(layer_inputs, verify)
        self.compute_eigenvalues()
        return self.max_eigenvalues



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
    max_eigenvalues = kfac.run(layer_inputs)
    pprint(max_eigenvalues)


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
 
   
     