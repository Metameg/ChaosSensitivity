import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import create_causal_mask
import dataclasses
from dataclasses import dataclass
import contextlib
from tqdm import tqdm

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

            

    


if __name__ == '__main__':
    prompt = "'This movie was terrible.'\n\n " \
    "Was the previous review positive or negative? " \
    "The previous review was "

    mmodel = MyModel()
    
    # Necessary to mimic a normal forward pass
    mmodel.model.config._attn_implementation = "eager"

    
    input_ids = mmodel.tokenize(prompt)
    seq_len = input_ids.shape[1]
    negative_ids = mmodel.tokenize(" negative")

    jacobians = []
    layers = mmodel.model.model.layers

    layer_inputs = mmodel.prepare_layer_inputs(input_ids)
    J_size = seq_len * mmodel.model.config.hidden_size
    chunk_size = 256


    # def forward_through_layer(h):
    #     print("Hidden size:", layer.hidden_size)
    #     print("self_attn", layer.self_attn)
    #     print("MLP", layer.mlp)
    #     print(mmodel.model.config._attn_implementation)

    #     out = layer(
    #         hidden_states=h,
    #         attention_mask=causal_mask,
    #         position_embeddings=position_embeddings,
    #         position_ids=position_ids,
    #         past_key_values=None,
    #         cache_position=cache_position
    #     )
    #     return out[0].reshape(-1) 
    
    for l, layer in enumerate(tqdm(layers)):
        print(f"\n=== Layer {l} ===")
        os.makedirs(f"jacobians/layer_{l}", exist_ok=True)

        for chunk_idx, output_start in enumerate(range(0, J_size, chunk_size)):

            output_end = min(output_start + chunk_size, J_size)
            h_in = layer_inputs.hidden_states.detach().requires_grad_(True)

            inputs_for_jacobian = dataclasses.replace(layer_inputs, hidden_states=h_in)

            def forward_partial(h):
                return mmodel.forward_layer(l, dataclasses.replace(inputs_for_jacobian, hidden_states=h), no_grad=False).hidden_states[0].reshape(-1)[output_start:output_end]

            J = torch.autograd.functional.jacobian(forward_partial, h_in)
            J = J.reshape(J_size, J_size)
            
            J_path = f"jacobians/layer_{l}.pt"
            torch.save(J.float().cpu(), J_path)
            
            
            
            print(J)
        
        # jacobians.append(J)

        layer_inputs = mmodel.forward_layer(l, layer_inputs)  # advance with no_grad
        # Advance h to the next layer's output (no grad needed)
        # with torch.no_grad():
           
        #     h = layer(
        #         h_in,
        #         attention_mask=causal_mask,
        #         position_embeddings=position_embeddings,
        #         cache_position=cache_position,
        #     )

        # Probe: project the last token's hidden state through norm + lm_head
        with torch.no_grad():
            last_token_h = layer_inputs.hidden_states[:, -1, :]           # shape: [1, hidden_size]
            normed = mmodel.norm(last_token_h)
            logits = mmodel.lm_head(normed)       # shape: [1, vocab_size]
            probs = torch.softmax(logits, dim=-1)

            top_k = 10
            top_probs, top_ids = torch.topk(probs[0], top_k)
            argmax_id = top_ids[0].item()
            argmax_token = mmodel.tokenizer.decode([argmax_id])
            print(f"\n--- After layer {l} ---")
            print(f"Argmax token: '{argmax_token}' (id={argmax_id}, prob={top_probs[0].item():.4f})")

        

            
        word_prob = 1.0
        for idx, token_id in enumerate(negative_ids[0,1:]):
            token_prob, argmax_id, argmax_prob = mmodel.probe_layer(input_ids, hl=l, token_id=token_id.item())

            word_prob *= token_prob
            # inputs = torch.cat([inputs, torch.tensor([[token_id]], device=inputs.device)], dim=1)

            # Print current token and its probability
            current_token = mmodel.tokenizer.decode(token_id)
            argmax_token = mmodel.tokenizer.decode(argmax_id)
            print("===Original====")
            print(
                # f"Target token: '{current_token}' | "
                f"P(target)={token_prob} | "
                f"Argmax: '{argmax_token}' | "
                f"P(argmax)={argmax_prob}"
            )
        
        print(f"Word probability up to this layer: {word_prob}")


    _, argmax_id, argmax_prob = mmodel.probe_layer(input_ids)
    
    
    final_token = mmodel.tokenizer.decode(argmax_id)


    print(f"\n=== Output (lm_head) ===")

    print(f"Most likely token: '{final_token}' | Probability: {argmax_prob}")
 
   
     