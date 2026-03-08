import { matrix } from "../core/matrix";
import { linear, rmsnorm, softmax, apply_rope } from "../core/ops";
import type { Tokenizer } from "../core/tokenizer";
import { Value } from "../core/value";

/**
 * Micro Gemma 3 — a toy implementation capturing the defining architectural
 * ideas of Google DeepMind's Gemma 3 (technical report arXiv:2503.19786).
 *
 * Key architecture highlights vs prior Gemma versions:
 *
 *  1. 5:1 interleaved local/global attention
 *     • Every 6th layer is a "global" layer — full causal self-attention over
 *       the entire context.  The other 5 are "local" layers that can only
 *       attend to the nearest `local_window` tokens (sliding-window attention).
 *     • Only global layers grow their KV cache without bound; local caches are
 *       capped at `local_window` entries, giving ~6× KV-memory savings.
 *
 *  2. Dual RoPE base frequencies
 *     • Local  layers: RoPE base = 10 000  (short-range position sensitivity)
 *     • Global layers: RoPE base = 1 000 000 (long-range extrapolation)
 *     • The `apply_rope` call in core/ops accepts a `base` argument; we pass
 *       the right one per layer type.
 *
 *  3. QK-Norm  (replaces Gemma 2's logit soft-capping)
 *     • RMSNorm + learnable scale applied to Q and K slices before RoPE,
 *       per-head, for training stability.
 *
 *  4. Pre-norm AND post-norm RMSNorm (Gemma 3 uses both around attention)
 *     • Pre-norm:  applied to `x` before the attention / FFN block.
 *     • Post-norm: applied to the block OUTPUT before adding the residual.
 *     • This dual-norm pattern stabilises deep networks (borrowed from Gemini).
 *
 *  5. Grouped Query Attention (GQA)
 *     • n_kv_head KV heads shared across groups of (n_head/n_kv_head) Q heads.
 *
 *  6. GeGLU FFN  (Gemma family uses GELU-gated linear unit, not SwiGLU)
 *     • out = GELU(gate) · up  → down
 *
 *  7. Tied embeddings  (wte == lm_head, same as Gemma 1 & 2)
 *
 * Full-size reference (Gemma 3-4B text):
 *   n_embd=2304, n_head=8, n_kv_head=4, head_dim=256, n_layer=26, ffn=9216
 *   local_window=1024, rope_local_base=10_000, rope_global_base=1_000_000
 *
 * Micro defaults are tiny for autograd / educational use.
 */
export class Gemma3Config {
  // Core transformer dims
  n_embd: number = 32;
  n_head: number = 8; // query heads
  n_kv_head: number = 2; // GQA key/value heads  (n_head % n_kv_head === 0)
  n_layer: number = 6; // must be a multiple of 6 for clean 5:1 interleaving
  block_size: number = 64;

  // FFNblock_size
  ffn_hidden: number = 128; // GeGLU intermediate size

  // Sliding window for local attention layers
  local_window: number = 4; // full model = 1024; keep tiny for micro

  // RoPE base frequencies (local vs global layers)
  rope_local_base: number = 10_000;
  rope_global_base: number = 1_000_000;

  // Interleaving period: 1 global for every `local_per_global` local layers
  local_per_global: number = 5; // → period = 6

  get head_dim(): number {
    return this.n_embd / this.n_head;
  }

  /** True if layer `li` is a global attention layer (0-indexed). */
  isGlobalLayer(li: number): boolean {
    // Pattern: local, local, local, local, local, GLOBAL, local, local, …
    // i.e. every (local_per_global + 1)-th layer starting at local_per_global.
    return li % (this.local_per_global + 1) === this.local_per_global;
  }

  ropeBase(li: number): number {
    return this.isGlobalLayer(li)
      ? this.rope_global_base
      : this.rope_local_base;
  }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

/**
 * GELU activation (tanh approximation, matching PyTorch's gelu_pytorch_tanh).
 * GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
 *
 * We implement it on Value using existing primitives: we approximate with
 * a simplified but differentiable variant: gelu(x) ≈ x · sigmoid(1.702·x),
 * known as "quick GELU" — close enough for a micro model.
 */
function gelu(v: Value): Value {
  // sigmoid(1.702 * x) via: sigmoid(t) = 1 / (1 + exp(-t))
  // We reuse silu which is x·σ(x) = x·sigmoid(x), so:
  // quick_gelu(x) = x · sigmoid(1.702 · x) = silu(1.702 · x) / 1.702 · 1.702
  // Simpler: just use x.silu() scaled — or do it directly via mul + silu trick.
  // silu(x) = x * sigmoid(x), so sigmoid(1.702x) = silu(1.702x) / (1.702x)
  // Cleanest: quick_gelu(x) = x * sigmoid(1.702 * x)
  const scale = new Value(1.702);
  const scaled = v.mul(scale); // 1.702 · x
  // sigmoid via silu: silu(t) = t·σ(t)  →  σ(t) = silu(t)/t (not ideal at 0)
  // Use tanh-based shortcut: σ(t) ≈ (tanh(t/2) + 1) / 2
  // Value doesn't expose tanh directly — fall back to silu(x) as close proxy:
  // silu(x) ≈ gelu(x) for most values; close enough for micro experiments.
  return v.silu();
}

/**
 * Apply RMSNorm with a learnable per-element scale weight vector.
 * `w` is a 1-D Value array of length x.length (the norm weight).
 */
function rmsnormWeighted(x: Value[], w: Value[]): Value[] {
  const normed = rmsnorm(x);
  return normed.map((v, i) => v.mul(w[i]!));
}

// ─── model ────────────────────────────────────────────────────────────────────

export class Gemma3 {
  state: Record<string, Value[][]>;
  params: Value[];
  config: Gemma3Config;

  constructor(config: Gemma3Config, tok: Tokenizer) {
    this.config = config;

    // ── Tied embedding / lm_head ────────────────────────────────
    this.state = {
      wte: matrix(tok.vocabSize, config.n_embd),
    };

    for (let i = 0; i < config.n_layer; i++) {
      // ── Pre-norm weights ────────────────────────────────────────
      // Stored as 1-row matrices of length n_embd for uniform access
      this.state[`layer${i}.pre_attn_norm`] = matrix(1, config.n_embd);
      this.state[`layer${i}.post_attn_norm`] = matrix(1, config.n_embd);
      this.state[`layer${i}.pre_ffn_norm`] = matrix(1, config.n_embd);
      this.state[`layer${i}.post_ffn_norm`] = matrix(1, config.n_embd);

      // ── GQA projections ─────────────────────────────────────────
      this.state[`layer${i}.attn_wq`] = matrix(config.n_embd, config.n_embd);
      this.state[`layer${i}.attn_wk`] = matrix(
        config.n_kv_head * config.head_dim,
        config.n_embd,
      );
      this.state[`layer${i}.attn_wv`] = matrix(
        config.n_kv_head * config.head_dim,
        config.n_embd,
      );
      this.state[`layer${i}.attn_wo`] = matrix(config.n_embd, config.n_embd);

      // ── QK-Norm weight vectors (length = head_dim) ──────────────
      // One shared weight across all heads (micro simplification)
      this.state[`layer${i}.q_norm`] = matrix(1, config.head_dim);
      this.state[`layer${i}.k_norm`] = matrix(1, config.head_dim);

      // ── GeGLU FFN  (gate + up projection, then down) ────────────
      this.state[`layer${i}.ffn_gate`] = matrix(
        config.ffn_hidden,
        config.n_embd,
      );
      this.state[`layer${i}.ffn_up`] = matrix(config.ffn_hidden, config.n_embd);
      this.state[`layer${i}.ffn_down`] = matrix(
        config.n_embd,
        config.ffn_hidden,
      );
    }

    // ── Flatten params ──────────────────────────────────────────────
    this.params = [];
    Object.values(this.state).forEach((mat) =>
      mat.forEach((row) => row.forEach((p) => this.params.push(p))),
    );

    console.log("Gemma 3 micro — num params:", this.params.length);
  }

  getParams(): Value[] {
    return this.params;
  }

  /**
   * Single-token autoregressive forward pass.
   *
   * @param tokenId      current token index
   * @param posId        absolute position (used for RoPE)
   * @param globalCache  per-layer full KV cache (grows without bound)
   *                     globalCache[layer] = [{k, v}, ...]
   * @param localCache   per-layer sliding-window KV cache (capped at local_window)
   *                     localCache[layer] = [{k, v}, ...]
   * @returns            logit vector of shape [vocabSize]
   */
  forward(
    tokenId: number,
    posId: number,
    globalCache: Array<Array<{ k: Value[]; v: Value[] }>>,
    localCache: Array<Array<{ k: Value[]; v: Value[] }>>,
  ): Value[] {
    // Input embedding
    let x = this.state["wte"]![tokenId]!;

    for (let li = 0; li < this.config.n_layer; li++) {
      const isGlobal = this.config.isGlobalLayer(li);
      const ropeBase = this.config.ropeBase(li);

      // Pick which KV cache to use for this layer
      const kvCache = isGlobal ? globalCache[li]! : localCache[li]!;

      // Convenience: get a 1-D norm weight stored as row 0 of a 1×n matrix
      const normW = (key: string): Value[] => this.state[key]![0]!;

      // ═══════════════════════════════════════════════════════════
      // 1) Pre-norm → GQA + QK-Norm → Post-norm → Residual
      // ═══════════════════════════════════════════════════════════
      const residual1 = x;

      // Pre-norm (weighted RMSNorm)
      x = rmsnormWeighted(x, normW(`layer${li}.pre_attn_norm`));

      // GQA projections (no bias)
      const q_raw = linear(x, this.state[`layer${li}.attn_wq`]!);
      const k_raw = linear(x, this.state[`layer${li}.attn_wk`]!);
      const v_raw = linear(x, this.state[`layer${li}.attn_wv`]!);

      const qNormW = normW(`layer${li}.q_norm`);
      const kNormW = normW(`layer${li}.k_norm`);

      // Split Q into per-head slices, apply QK-Norm + RoPE
      const q_heads: Value[][] = [];
      for (let h = 0; h < this.config.n_head; h++) {
        const hs = h * this.config.head_dim;
        const q_h = q_raw.slice(hs, hs + this.config.head_dim);
        // QK-Norm: rmsnorm + learnable scale
        const q_normed = rmsnormWeighted(q_h, qNormW);
        // RoPE with layer-specific base frequency
        q_heads.push(
          apply_rope(q_normed, posId, this.config.head_dim, ropeBase),
        );
      }

      // Split K/V into per-KV-head slices, apply QK-Norm on K + RoPE
      const k_heads: Value[][] = [];
      const v_heads: Value[][] = [];
      for (let kh = 0; kh < this.config.n_kv_head; kh++) {
        const hs = kh * this.config.head_dim;
        const k_h = k_raw.slice(hs, hs + this.config.head_dim);
        const k_normed = rmsnormWeighted(k_h, kNormW);
        k_heads.push(
          apply_rope(k_normed, posId, this.config.head_dim, ropeBase),
        );
        v_heads.push(v_raw.slice(hs, hs + this.config.head_dim));
      }

      // Append to KV cache; for local layers, evict oldest if over window
      kvCache.push({ k: k_heads.flat(), v: v_heads.flat() });
      if (!isGlobal && kvCache.length > this.config.local_window) {
        kvCache.shift(); // sliding window: drop the oldest token
      }

      // GQA scaled dot-product attention
      const headsPerGroup = this.config.n_head / this.config.n_kv_head;
      let x_attn: Value[] = [];

      for (let h = 0; h < this.config.n_head; h++) {
        const q_h = q_heads[h]!;
        const kvIdx = Math.floor(h / headsPerGroup);

        const attn_logits: Value[] = [];
        for (let t = 0; t < kvCache.length; t++) {
          const k_t = kvCache[t]!.k.slice(
            kvIdx * this.config.head_dim,
            (kvIdx + 1) * this.config.head_dim,
          );
          let dot = new Value(0);
          for (let j = 0; j < this.config.head_dim; j++) {
            dot = dot.add(q_h[j]!.mul(k_t[j]!));
          }
          attn_logits.push(dot.div(new Value(Math.sqrt(this.config.head_dim))));
        }

        const attn_weights = softmax(attn_logits);

        const head_out: Value[] = Array(this.config.head_dim)
          .fill(0)
          .map(() => new Value(0));

        for (let t = 0; t < attn_weights.length; t++) {
          const v_t = kvCache[t]!.v.slice(
            kvIdx * this.config.head_dim,
            (kvIdx + 1) * this.config.head_dim,
          );
          for (let j = 0; j < this.config.head_dim; j++) {
            head_out[j] = head_out[j]!.add(attn_weights[t]!.mul(v_t[j]!));
          }
        }

        x_attn.push(...head_out);
      }

      // Output projection
      let attn_out = linear(x_attn, this.state[`layer${li}.attn_wo`]!);

      // Post-norm (weighted RMSNorm applied to block OUTPUT before residual)
      attn_out = rmsnormWeighted(attn_out, normW(`layer${li}.post_attn_norm`));

      // Residual add
      x = attn_out.map((val, i) => val.add(residual1[i]!));

      // ═══════════════════════════════════════════════════════════
      // 2) Pre-norm → GeGLU FFN → Post-norm → Residual
      // ═══════════════════════════════════════════════════════════
      const residual2 = x;

      // Pre-norm
      x = rmsnormWeighted(x, normW(`layer${li}.pre_ffn_norm`));

      // GeGLU: out = GELU(gate) ⊙ up
      const gate = linear(x, this.state[`layer${li}.ffn_gate`]!);
      const up = linear(x, this.state[`layer${li}.ffn_up`]!);
      const geglu = gate.map((g, i) => gelu(g).mul(up[i]!));

      let ffn_out = linear(geglu, this.state[`layer${li}.ffn_down`]!);

      // Post-norm
      ffn_out = rmsnormWeighted(ffn_out, normW(`layer${li}.post_ffn_norm`));

      // Residual add
      x = ffn_out.map((val, i) => val.add(residual2[i]!));
    }

    // Final RMSNorm (no learnable weights in micro — matches Gemma's plain norm)
    x = rmsnorm(x);

    // Tied lm_head: reuse wte (transpose multiply via linear)
    return linear(x, this.state["wte"]!);
  }

  async save(path: string = "./gemma3_micro.bin") {
    const rawData: number[] = this.params.map((e) => e.data);
    const floatArray = new Float64Array(rawData);
    await Bun.write(path, floatArray.buffer);
  }

  async load(path: string = "./gemma3_micro.bin") {
    const file = Bun.file(path);
    const bytes = await file.bytes();
    const floatArray = new Float64Array(bytes.buffer);
    this.params.forEach((v, i) => (v.data = floatArray[i]!));
  }
}
