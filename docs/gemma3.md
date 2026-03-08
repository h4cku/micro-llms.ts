# Gemma 3 Architecture Summary

Gemma 3 is a micro-implementation inspired by Google DeepMind's Gemma 3. Key architectural features include:

1.  **5:1 Interleaved Local/Global Attention:**
    *   Every 6th layer employs "global" causal self-attention over the entire context.
    *   The other 5 layers use "local" attention, limited to a `local_window` of tokens (sliding-window attention).
    *   Global layers have an unbounded KV cache, while local caches are capped, saving KV memory.
2.  **Dual RoPE Base Frequencies:**
    *   Local layers use a RoPE base of 10,000 for short-range sensitivity.
    *   Global layers use a RoPE base of 1,000,000 for long-range extrapolation.
3.  **QK-Norm:** RMSNorm with a learnable scale is applied to Query (Q) and Key (K) slices before RoPE, per-head, for training stability.
4.  **Pre-norm AND Post-norm RMSNorm:** Both are used around attention and FFN blocks, enhancing stability in deep networks.
5.  **Grouped Query Attention (GQA):** `n_kv_head` Key/Value (KV) heads are shared across groups of Query (Q) heads.
6.  **GeGLU FFN:** Uses a GELU-gated linear unit for the Feed-Forward Network.
7.  **Tied Embeddings:** The `wte` (word token embedding) and `lm_head` (language model head) weights are shared.

## Implementation Details

*   The `Gemma3Config` class defines parameters like `n_embd`, `n_head`, `n_kv_head`, `n_layer`, `block_size`, `ffn_hidden`, `local_window`, and the dual RoPE bases. It also includes helper methods `isGlobalLayer` and `ropeBase` to determine layer type and RoPE base.
*   The `gelu` helper function implements a "quick GELU" approximation using `silu`.
*   `rmsnormWeighted` applies RMSNorm with a learnable per-element scale.
*   The `Gemma3` class initializes the model state with tied embeddings and various weights for attention (WQ, WK, WV, WO), QK-Norm (q_norm, k_norm), and FFN (ffn_gate, ffn_up, ffn_down), along with pre/post normalization weights.
*   The `forward` method processes tokens layer by layer, managing separate `globalCache` and `localCache` for KV pairs based on whether a layer is global or local. It applies pre-norm, GQA with QK-Norm and RoPE (with appropriate base), post-norm, residual connections, and then the GeGLU FFN with its own pre/post-norm and residual. Finally, a final RMSNorm and linear projection using the tied `lm_head` (wte) are applied.
*   `save` and `load` methods handle model parameter serialization and deserialization using `Bun.write` and `Bun.file`.