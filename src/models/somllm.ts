import { matrix } from "../core/matrix";
import { linear, rmsnorm, softmax, apply_rope } from "../core/ops";
import type { Tokenizer } from "../core/tokenizer";
import { Value } from "../core/value";

/**
 * Micro SmolLM3 — a toy implementation capturing the key architectural ideas
 * of HuggingFace's SmolLM3 (July 2025):
 *
 *  • Grouped Query Attention (GQA)  — n_head query heads, n_kv_head KV heads
 *  • NoPE                           — every 4th layer skips RoPE entirely
 *  • SwiGLU FFN                     — gate · silu(up) → down  (no MoE)
 *  • Tied embeddings                — wte is reused as lm_head
 *  • RMSNorm pre-norm on every block
 *
 * Full-size reference dimensions (3B):
 *   n_embd=2048, n_head=16, n_kv_head=4, n_layer=36, ffn_mult=4
 *
 * Micro defaults are intentionally tiny so the whole thing can be trained
 * in a browser / Bun script with autograd.
 */
export class SmolLM3Config {
  // Core transformer dims
  n_embd: number = 32; // hidden size
  n_head: number = 8; // query heads
  n_kv_head: number = 2; // key/value heads  (GQA: n_head must be divisible by n_kv_head)
  n_layer: number = 4; // transformer blocks
  block_size: number = 64; // max sequence length

  // FFN
  ffn_mult: number = 4; // intermediate = ffn_mult * n_embd  (SwiGLU: gate + up + down)

  // NoPE: layer indices where RoPE is disabled (every 4th layer, 0-indexed)
  // e.g. for n_layer=4: layer 3 is NoPE
  nope_period: number = 4;

  // Derived helpers
  get head_dim(): number {
    return this.n_embd / this.n_head;
  }

  get ffn_hidden(): number {
    return this.ffn_mult * this.n_embd;
  }

  /** True if this layer index should skip RoPE (NoPE) */
  isNoPeLayer(layerIdx: number): boolean {
    // Every `nope_period`-th layer (0-indexed), starting from nope_period-1
    return (layerIdx + 1) % this.nope_period === 0;
  }
}

export class SmolLM3 {
  state: Record<string, Value[][]>;
  params: Value[];
  config: SmolLM3Config;

  constructor(config: SmolLM3Config, tok: Tokenizer) {
    this.config = config;

    // ---------- Embedding (tied with lm_head) ----------
    this.state = {
      wte: matrix(tok.vocabSize, config.n_embd),
    };

    for (let i = 0; i < config.n_layer; i++) {
      // ---- Grouped Query Attention ----
      // Q projection: n_embd → n_embd  (n_head * head_dim)
      this.state[`layer${i}.attn_wq`] = matrix(config.n_embd, config.n_embd);

      // K projection: n_embd → n_kv_head * head_dim
      this.state[`layer${i}.attn_wk`] = matrix(
        config.n_kv_head * config.head_dim,
        config.n_embd,
      );

      // V projection: n_embd → n_kv_head * head_dim
      this.state[`layer${i}.attn_wv`] = matrix(
        config.n_kv_head * config.head_dim,
        config.n_embd,
      );

      // Output projection: n_embd → n_embd
      this.state[`layer${i}.attn_wo`] = matrix(config.n_embd, config.n_embd);

      // ---- SwiGLU FFN ----
      // gate: n_embd → ffn_hidden
      this.state[`layer${i}.ffn_gate`] = matrix(
        config.ffn_hidden,
        config.n_embd,
      );

      // up:   n_embd → ffn_hidden
      this.state[`layer${i}.ffn_up`] = matrix(config.ffn_hidden, config.n_embd);

      // down: ffn_hidden → n_embd
      this.state[`layer${i}.ffn_down`] = matrix(
        config.n_embd,
        config.ffn_hidden,
      );
    }

    // Flatten all parameters
    this.params = [];
    Object.values(this.state).forEach((mat) =>
      mat.forEach((row) => row.forEach((p) => this.params.push(p))),
    );

    console.log("SmolLM3 micro — num params:", this.params.length);
  }

  getParams(): Value[] {
    return this.params;
  }

  /**
   * Single-token forward pass (autoregressive / KV-cache style).
   *
   * @param tokenId   current token index
   * @param posId     position index (used for RoPE)
   * @param kvCache   per-layer KV cache: kvCache[layer] = [{k, v}, ...]
   * @returns         logit vector of shape [vocabSize]
   */
  forward(
    tokenId: number,
    posId: number,
    kvCache: Array<Array<{ k: Value[]; v: Value[] }>>,
  ): Value[] {
    let x = this.state["wte"]![tokenId]!;
    x = rmsnorm(x);

    for (let li = 0; li < this.config.n_layer; li++) {
      const useRoPE = !this.config.isNoPeLayer(li);

      // =========================================================
      // 1) Grouped Query Attention (GQA) with optional NoPE
      // =========================================================
      const residual1 = x;
      x = rmsnorm(x);

      // --- Projections ---
      const q_raw = linear(x, this.state[`layer${li}.attn_wq`]!); // [n_embd]
      const k_raw = linear(x, this.state[`layer${li}.attn_wk`]!); // [n_kv_head * head_dim]
      const v_raw = linear(x, this.state[`layer${li}.attn_wv`]!); // [n_kv_head * head_dim]

      // --- Split K/V into per-KV-head slices ---
      const k_heads: Value[][] = [];
      const v_heads: Value[][] = [];
      for (let kh = 0; kh < this.config.n_kv_head; kh++) {
        const hs = kh * this.config.head_dim;
        const k_h = k_raw.slice(hs, hs + this.config.head_dim);
        const v_h = v_raw.slice(hs, hs + this.config.head_dim);

        // Apply RoPE to K only when this is not a NoPE layer
        k_heads.push(
          useRoPE ? apply_rope(k_h, posId, this.config.head_dim) : k_h,
        );
        v_heads.push(v_h);
      }

      // --- Store in KV cache ---
      kvCache[li]!.push({ k: k_heads.flat(), v: v_heads.flat() });

      // --- Split Q into per-query-head slices ---
      const q_heads: Value[][] = [];
      for (let h = 0; h < this.config.n_head; h++) {
        const hs = h * this.config.head_dim;
        const q_h = q_raw.slice(hs, hs + this.config.head_dim);
        q_heads.push(
          useRoPE ? apply_rope(q_h, posId, this.config.head_dim) : q_h,
        );
      }

      // --- GQA: each query group shares one KV head ---
      const headsPerGroup = this.config.n_head / this.config.n_kv_head;
      let x_attn: Value[] = [];

      for (let h = 0; h < this.config.n_head; h++) {
        const q_h = q_heads[h]!;
        // Which KV head does this query head belong to?
        const kvIdx = Math.floor(h / headsPerGroup);

        const attn_logits: Value[] = [];

        for (let t = 0; t < kvCache[li]!.length; t++) {
          // Extract the relevant KV head slice from the cache
          const k_t = kvCache[li]![t]!.k.slice(
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
          const v_t = kvCache[li]![t]!.v.slice(
            kvIdx * this.config.head_dim,
            (kvIdx + 1) * this.config.head_dim,
          );
          for (let j = 0; j < this.config.head_dim; j++) {
            head_out[j] = head_out[j]!.add(attn_weights[t]!.mul(v_t[j]!));
          }
        }

        x_attn.push(...head_out);
      }

      // Output projection + residual
      x = linear(x_attn, this.state[`layer${li}.attn_wo`]!);
      x = x.map((val, i) => val.add(residual1[i]!));

      // =========================================================
      // 2) SwiGLU FFN  (gate · silu(up) → down)
      // =========================================================
      const residual2 = x;
      x = rmsnorm(x);

      const gate = linear(x, this.state[`layer${li}.ffn_gate`]!);
      const up = linear(x, this.state[`layer${li}.ffn_up`]!);

      // SwiGLU: element-wise gate * silu(up)
      const swiglu: Value[] = gate.map((g, i) => g.mul(up[i]!.silu()));

      const ffn_out = linear(swiglu, this.state[`layer${li}.ffn_down`]!);

      x = ffn_out.map((val, i) => val.add(residual2[i]!));
    }

    // Final norm
    x = rmsnorm(x);

    // Tied embedding: lm_head reuses wte weights (transpose multiply)
    return linear(x, this.state["wte"]!);
  }

  async save(path: string = "./smollm3_micro.bin") {
    const rawData: number[] = this.params.map((e) => e.data);
    const floatArray = new Float64Array(rawData);
    await Bun.write(path, floatArray.buffer);
  }

  async load(path: string = "./smollm3_micro.bin") {
    const file = Bun.file(path);
    const bytes = await file.bytes();
    const floatArray = new Float64Array(bytes.buffer);
    this.params.forEach((v, i) => (v.data = floatArray[i]!));
  }
}
