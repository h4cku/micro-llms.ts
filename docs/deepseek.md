# Deepseek Architecture Summary

Deepseek is a micro-implementation of the Deepseek LLM. Its key architectural features are:

1.  **Multi-Head Latent Attention (MLA):**
    *   Compresses Key/Value (KV) pairs into a lower `latent_dim` representation (`attn_compress_kv`).
    *   Uses this compressed latent KV in the attention mechanism to generate full KV for dot-product attention.
    *   Applies Rotational Positional Embeddings (RoPE) to Query (Q) and Key (K) slices.
2.  **Mixture of Experts (MoE):**
    *   Employs `n_experts` feed-forward experts.
    *   A `router` network determines the contribution of each expert via `softmax` probabilities.
    *   Each expert consists of a `gate`, `up`, and `down` linear layers, where `gate` and `up` outputs are combined using a Swish-Gated Linear Unit (SwiGLU) like activation (`silu`).
    *   The outputs of all experts are weighted by the router probabilities and summed.
3.  **RMSNorm:** Applied before both the MLA and MoE blocks, and as a final normalization layer.
4.  **Tied Embeddings:** The `wte` (word token embedding) and `lm_head` (language model head) weights are separate projections.

## Implementation Details:

*   The `DeepseekConfig` class defines core transformer dimensions (`n_embd`, `n_head`, `n_layer`, `block_size`), and DeepSeek-specific parameters like `latent_dim` (MLA compression dimension), `n_experts` (number of MoE experts), and `n_active` (number of active experts, though noted as "not used in micro version (soft routing)").
*   The `Deepseek` class initializes `wte` and `lm_head` embeddings.
*   Layer initialization includes weights for MLA components (attn_wq, attn_compress_kv, attn_wkv, attn_wo) and MoE components (router, and for each expert: expert_gate, expert_up, expert_down).
*   The `forward` method processes tokens layer by layer. It applies RMSNorm, then the MLA block:
    *   Generates Q, compresses KV into `kv_latent` and stores it in `keysLatentCache`.
    *   Splits Q into heads and applies RoPE.
    *   For attention, it reconstructs full KV from latent KV in the cache, applies RoPE to K, calculates dot-product logits, applies softmax, and computes the weighted sum of V.
    *   After MLA, a residual connection is added.
*   Then, the MoE block:
    *   Calculates `router_logits`, applies `softmax` to get `router_probs`.
    *   For each expert, it calculates `gate`, `up`, forms a SwiGLU-like activation, and projects to `out`.
    *   Expert outputs are weighted by `router_probs` and summed.
    *   Another residual connection is added.
*   A final RMSNorm and linear projection using `lm_head` produce the final logits.
*   `save` and `load` methods handle model parameter serialization and deserialization using `Bun.write` and `Bun.file`.