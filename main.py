import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import create_causal_mask

from tqdm import tqdm

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
    
    def _probe_forward(self, inputs, hl):
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
            logits = self.lm_head(self.norm(outputs.hidden_states[hl]))
            probs = torch.softmax(logits, dim=-1)

        return probs
    
      
    

    def jacobian(self, inputs):
        # Get the input embeddings
        embeddings = self.model.model.embed_tokens(inputs)  # [1, seq_len, d_model]
        embeddings = embeddings.detach().requires_grad_(True)

        J = torch.autograd.functional.jacobian(self._forward_to_layer, embeddings)
        # shape: [seq_len*d_model, 1, seq_len*d_model] → squeeze

        return J[:, 0, :, :]  # [seq_len*d_model, seq_len*d_model] ... needs reshape
        

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
    
    # def early_exit(threshold):
            

    


if __name__ == '__main__':
    prompt = "'This movie was terrible.'\n\n " \
    "Was the previous review positive or negative? " \
    "The previous review was "

    mmodel = MyModel()
    print("Before:", mmodel.model.config._attn_implementation)

    mmodel.model.config._attn_implementation = "eager"

    print("After:", mmodel.model.config._attn_implementation)
    inputs = mmodel.tokenize(prompt)
    negative_ids = mmodel.tokenize(" negative")

    jacobians = []
    layers = mmodel.model.model.layers

    

    
    h = mmodel.model.model.embed_tokens(inputs)

    

    # Also need position ids and causal mask for the attention layers (rotary_emb doesn't handle pos_ids automatically)
    seq_len = inputs.shape[1]
    position_ids = torch.arange(seq_len, device=inputs.device).unsqueeze(0)

    cache_position = torch.arange(seq_len, device=inputs.device)


    causal_mask = create_causal_mask(
        config=mmodel.model.config,
        input_embeds=h,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    position_embeddings = mmodel.model.model.rotary_emb(
        h,
        position_ids=position_ids
    )


    def forward_through_layer(h):
        print("Hidden size:", layer.hidden_size)
        print("self_attn", layer.self_attn)
        print("MLP", layer.mlp)
        print(mmodel.model.config._attn_implementation)

        out = layer(
            hidden_states=h,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position
        )
        return out[0].reshape(-1) 
    
    for l, layer in enumerate(tqdm(layers)):
        print(f"\n=== Layer {l} ===")

        h_in = h.detach().requires_grad_(True)

        J = torch.autograd.functional.jacobian(forward_through_layer, h_in)

        J = J.reshape(seq_len * mmodel.model.config.hidden_size,
                      seq_len * mmodel.model.config.hidden_size)
        jacobians.append(J)

        # Advance h to the next layer's output (no grad needed)
        with torch.no_grad():
            position_embeddings = mmodel.model.model.rotary_emb(
                h_in,
                position_ids=position_ids
            )

            h = layer(
                h_in,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )

            
        # word_prob = 1.0
        # for idx, token_id in enumerate(negative_ids[0,1:]):
        #     token_prob, argmax_id, argmax_prob = mmodel.probe_layer(inputs, hl=l, token_id=token_id.item())

        #     word_prob *= token_prob
        #     # inputs = torch.cat([inputs, torch.tensor([[token_id]], device=inputs.device)], dim=1)

        #     # Print current token and its probability
        #     current_token = mmodel.tokenizer.decode(token_id)
        #     argmax_token = mmodel.tokenizer.decode(argmax_id)

        #     print(
        #         # f"Target token: '{current_token}' | "
        #         f"P(target)={token_prob} | "
        #         f"Argmax: '{argmax_token}' | "
        #         f"P(argmax)={argmax_prob}"
        #     )
        
        # print(f"Word probability up to this layer: {word_prob}")

    _, argmax_id, argmax_prob = mmodel.probe_layer(inputs)
    
    
    final_token = mmodel.tokenizer.decode(argmax_id)


    print(f"\n=== Output (lm_head) ===")

    print(f"Most likely token: '{final_token}' | Probability: {argmax_prob}")
 
   
     