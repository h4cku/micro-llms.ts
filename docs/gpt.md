# GPT Architecture Summary

This is a micro-implementation of a GPT-like architecture, often referred to as a "decoder-only transformer." Key architectural features include:

1.  **Decoder-Only Transformer Blocks:** Each layer consists of a multi-head self-attention mechanism followed by a two-layer Feed-Forward Network (FFN).
2.  **Multi-Head Self-Attention:**
    *   Queries (Q), Keys (K), and Values (V) are projected from the input.
    *   For each head, attention logits are calculated as a dot product of Q and K, scaled by the square root of the head dimension.
    *   `softmax` is applied to the logits to obtain attention weights.
    *   The output of each head is a weighted sum of V.
    *   Outputs from all heads are concatenated and then linearly projected.
3.  **Positional Embeddings:** Both Word Token Embeddings (`wte`) and Word Positional Embeddings (`wpe`) are used. The input to the transformer blocks is the sum of these two embeddings.
4.  **Feed-Forward Network (FFN):** A two-layer MLP (`mlp_fc1` and `mlp_fc2`) with a non-linear activation function (ReLU squared in this micro-implementation: `relu().pow(2)`).
5.  **RMSNorm:** Applied at the beginning of each transformer block (before attention and FFN) and after the input embedding.
6.  **Residual Connections:** Applied after the attention block and after the FFN block.
7.  **Separate Embeddings:** `wte`, `wpe` and `lm_head` are distinct projections.

## Implementation Details:

*   The `GPTConfig` class defines basic transformer dimensions (`n_embd`, `n_head`, `n_layer`, `block_size`) and calculates `head_dim`.
*   The `GPT` class initializes the model state with `wte`, `wpe` (positional embedding), and `lm_head` (language model head).
*   For each layer, it initializes weights for attention projections (attn_wq, attn_wk, attn_wv, attn_wo) and FFN layers (mlp_fc1, mlp_fc2).
*   The `forward` method takes a `tokenId` and `posId`. It combines `wte` and `wpe` for the initial input.
*   Each layer performs:
    *   RMSNorm.
    *   Linear projections for Q, K, V.
    *   Stores K and V in `keys` and `values` caches.
    *   Performs multi-head attention: slices Q, K, V for each head, calculates scaled dot-product attention, applies softmax, and computes weighted sum of values.
    *   Linearly projects concatenated head outputs.
    *   Adds a residual connection.
    *   RMSNorm.
    *   Passes through the FFN with ReLU squared activation.
    *   Adds another residual connection.
*   Finally, the output is passed through a linear layer (`lm_head`) to produce logits.
*   `save` and `load` methods handle model parameter serialization and deserialization using `Bun.write` and `Bun.file`.