# SmolLM3 Architecture Summary

SmolLM3 is a micro-implementation inspired by HuggingFace's SmolLM3. Key architectural features include:

1.  **Grouped Query Attention (GQA):** Uses `n_head` query heads and `n_kv_head` KV heads, where query head groups share KV heads.
2.  **NoPE (No Positional Embeddings):** Every `nope_period`-th layer (e.g., every 4th layer) skips Rotational Positional Embeddings (RoPE) entirely. For other layers, RoPE is applied to Q and K.
3.  **SwiGLU FFN:** Employs a Swish Gated Linear Unit (SwiGLU) Feed-Forward Network with the structure `gate · silu(up) → down` (no Mixture of Experts).
4.  **Tied Embeddings:** The `wte` (word token embedding) is reused as the `lm_head` (language model head).
5.  **RMSNorm pre-norm:** RMSNorm is applied before each attention and FFN block.

## Implementation Details:

*   The `SmolLM3Config` class defines core transformer dimensions (`n_embd`, `n_head`, `n_kv_head`, `n_layer`, `block_size`), FFN multiplier (`ffn_mult`), and the `nope_period` for disabling RoPE. It includes a helper `isNoPeLayer` to determine if a layer should skip RoPE.
*   The `SmolLM3` class initializes the model state with tied embeddings (`wte` also serving as `lm_head`).
*   Layer initialization includes weights for GQA projections (WQ, WK, WV, WO) and SwiGLU FFN (ffn_gate, ffn_up, ffn_down).
*   The `forward` method processes tokens layer by layer. It applies pre-norm RMSNorm, then performs GQA. During GQA, it checks `isNoPeLayer` to conditionally apply RoPE to Q and K slices. The KV cache is managed per layer. After GQA, output projection and a residual connection are applied. Then, another pre-norm RMSNorm is applied before the SwiGLU FFN. Finally, a residual connection is added after the FFN. A final RMSNorm and linear projection using the tied `lm_head` complete the forward pass.
*   `save` and `load` methods handle model parameter serialization and deserialization using `Bun.write` and `Bun.file`.