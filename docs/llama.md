# Llama Architecture Summary

This is a micro-implementation of the Llama-like architecture. Key architectural features include:

1.  **Grouped Query Attention (GQA):** Uses `n_head` query heads and `n_kv_head` key/value heads. Query heads are grouped, and each group shares a single KV head (controlled by `n_rep`).
2.  **Rotational Positional Embeddings (RoPE):** Applied to both Query (Q) and Key (K) slices before attention calculation.
3.  **SwiGLU FFN:** Employs a Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network: `gate · silu(up) → down`.
4.  **RMSNorm:** Applied at the beginning of each transformer block (before attention and FFN) and as a final normalization layer.
5.  **Residual Connections:** Applied after the attention block and after the FFN block.
6.  **Separate Embeddings:** `wte` (word token embedding) and `lm_head` (language model head) are distinct projections.

## Implementation Details:

*   The `LlamaConfig` class defines core transformer dimensions (`n_embd`, `n_head`, `n_kv_head`, `n_layer`, `block_size`) and calculates `head_dim` and `n_rep` (number of query heads per KV head).
*   The `Llama` class initializes the model state with `wte` and `lm_head` embeddings.
*   Layer initialization includes weights for attention projections (attn_wq, attn_wk, attn_wv, attn_wo) and SwiGLU FFN layers (ffn_gate, ffn_up, ffn_down).
*   The `forward` method takes a `tokenId` and `posId`. It starts by applying RMSNorm to the `wte` output.
*   Each layer performs:
    *   RMSNorm.
    *   Linear projections for Q, K, V.
    *   Splits Q, K, V into respective heads and applies RoPE to Q and K.
    *   Stores `k_heads` and `v_heads` in `keys` and `values` caches.
    *   Performs GQA: for each query head, it uses the corresponding shared KV head from the cache, calculates scaled dot-product attention, applies softmax, and computes the weighted sum of values.
    *   Linearly projects concatenated head outputs.
    *   Adds a residual connection.
    *   RMSNorm.
    *   Passes through the SwiGLU FFN.
    *   Adds another residual connection.
*   Finally, a final RMSNorm and a linear projection using `lm_head` produce the final logits.
*   `save` and `load` methods handle model parameter serialization and deserialization using `Bun.write` and `Bun.file`.