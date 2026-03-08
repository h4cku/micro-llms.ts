# Qwen 3 Architecture Summary

Qwen 3 is a micro-implementation of Alibaba's Qwen3 series. It supports both dense and Mixture of Experts (MoE) variants.

## Dense Variant Architecture Highlights:

1.  **Grouped Query Attention (GQA):** Employs `n_head` query heads and `n_kv_head` KV heads.
2.  **QK-Norm:** RMSNorm is applied to Query (Q) and Key (K) slices before the dot-product, enhancing training stability.
3.  **No QKV Bias:** Qwen3 removed QKV bias compared to Qwen2.
4.  **RoPE on every layer:** Rotational Positional Embeddings are applied in every layer.
5.  **SwiGLU FFN:** Uses a SwiGLU (Swish Gated Linear Unit) Feed-Forward Network: `gate · silu(up) → down`.
6.  **Pre-norm RMSNorm:** RMSNorm is applied before each attention and FFN block.

## MoE Variant Additions (activated with `config.use_moe = true`):

*   **`n_experts` FFN experts per layer:** Allows for fine-grained expert segmentation.
*   **Top-k Routing:** Utilizes `softmax` and `argTopK` for routing, selecting `n_active` experts per token (no shared experts).
*   **Load-balancing auxiliary loss signal:** Router probabilities (`router_probs`) contribute to a load-balancing loss during training.

## Thinking-mode Flag (activated with `config.thinking_mode = true`):

*   Introduces a learnable scalar "thinking_gate" per layer that scales the FFN output, serving as a micro-proxy for a budget mechanism.

## Implementation Details:

*   The `Qwen3Config` class defines core transformer dimensions (`n_embd`, `n_head`, `n_kv_head`, `n_layer`, `block_size`), FFN multiplier (`ffn_mult`), and optional MoE (`use_moe`, `n_experts`, `n_active`) and `thinking_mode` parameters.
*   Helper functions `qknorm` (RMSNorm for QK-Norm) and `topKIndices` (for selecting top-k experts) are provided.
*   The `Qwen3` class initializes `wte` (word token embedding) and `lm_head` (language model head) as *separate* embeddings (not tied).
*   Layer initialization includes attention projections (WQ, WK, WV, WO), QK-Norm weights (q_norm, k_norm).
*   Depending on `config.use_moe`, it initializes either dense SwiGLU FFN weights (ffn_gate, ffn_up, ffn_down) or MoE components (router and per-expert gate, up, down weights).
*   If `config.thinking_mode` is true, a `think_gate` scalar is added per layer.
*   The `forward` method applies pre-norm RMSNorm, GQA with QK-Norm and RoPE, and residual connections. It then proceeds to either the dense SwiGLU FFN or the sparse MoE FFN, where for MoE, it calculates router logits, softmax probabilities, selects top-k experts, renormalizes weights, and accumulates weighted expert outputs. The thinking-mode gate is applied if active. Finally, a final RMSNorm and separate `lm_head` projection are performed.
*   `save` and `load` methods handle model parameter serialization and deserialization using `Bun.write` and `Bun.file`.