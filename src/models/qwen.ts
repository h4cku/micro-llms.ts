import { matrix } from "../core/matrix";
import { linear, rmsnorm, softmax, apply_rope } from "../core/ops";
import type { Tokenizer } from "../core/tokenizer";
import { Value } from "../core/value";

/**
 * Micro Qwen3 — a toy implementation capturing the key architectural ideas
 * of Alibaba's Qwen3 series (May 2025 technical report, arxiv:2505.09388).
 *
 * Dense variant architecture:
 *   • Grouped Query Attention (GQA)  — n_head query heads, n_kv_head KV heads
 *   • QK-Norm                        — RMSNorm applied to Q and K before dot-product
 *                                      (new in Qwen3; stabilises training at scale)
 *   • No QKV bias                    — removed vs Qwen2
 *   • RoPE on every layer            — no NoPE
 *   • SwiGLU FFN                     — gate · silu(up) → down  (dense path)
 *   • Pre-norm RMSNorm on each block
 *
 * MoE variant adds (enable with config.use_moe = true):
 *   • n_experts FFN experts per layer (fine-grained expert segmentation)
 *   • top-k routing via softmax + argTopK (no shared experts, unlike Qwen2.5-MoE)
 *   • Load-balancing auxiliary loss signal via router_probs (training side)
 *
 * Thinking-mode flag:
 *   • config.thinking_mode adds a learnable scalar "thinking_gate" per layer
 *     that scales the FFN output — a micro-proxy for the budget mechanism.
 *
 * Full-size reference dimensions (Qwen3-8B dense):
 *   n_embd=4096, n_head=32, n_kv_head=8, n_layer=36, ffn_mult≈2.625 (11008 hidden)
 *
 * Micro defaults are intentionally tiny for autograd experiments.
 */
export class Qwen3Config {
  // Core transformer dims
  n_embd: number = 32;
  n_head: number = 8; // query heads
  n_kv_head: number = 2; // GQA: key/value heads  (n_head % n_kv_head === 0)
  n_layer: number = 4;
  block_size: number = 64;

  // FFN
  ffn_mult: number = 4; // ffn_hidden = ffn_mult * n_embd

  // MoE (optional — set use_moe=true to activate)
  use_moe: boolean = false;
  n_experts: number = 8; // total experts per MoE layer  (full: 128)
  n_active: number = 2; // top-k active experts per token  (full: 8)

  // Thinking mode: adds a per-layer learnable FFN gate scalar
  thinking_mode: boolean = false;

  // Derived helpers
  get head_dim(): number {
    return this.n_embd / this.n_head;
  }

  get ffn_hidden(): number {
    return this.ffn_mult * this.n_embd;
  }
}

// ─── small helpers ────────────────────────────────────────────────────────────

/** RMSNorm over a 1-D Value array (used for QK-Norm) */
function qknorm(x: Value[]): Value[] {
  return rmsnorm(x);
}

/**
 * Top-k indices of a Value array (returns plain number indices so we can
 * hard-select experts without differentiating through the sort).
 * Gradients flow through the router_probs weights instead.
 */
function topKIndices(probs: Value[], k: number): number[] {
  const pairs = probs.map((p, i) => ({ val: p.data, i }));
  pairs.sort((a, b) => b.val - a.val);
  return pairs.slice(0, k).map((p) => p.i);
}

// ─── model ────────────────────────────────────────────────────────────────────

export class Qwen3 {
  state: Record<string, Value[][]>;
  // Scalar params live outside the matrix map (thinking gates)
  thinkingGates: Value[];
  params: Value[];
  config: Qwen3Config;

  constructor(config: Qwen3Config, tok: Tokenizer) {
    this.config = config;
    this.thinkingGates = [];

    // ── Embeddings (NOT tied — Qwen3 uses separate lm_head) ──
    this.state = {
      wte: matrix(tok.vocabSize, config.n_embd),
      lm_head: matrix(tok.vocabSize, config.n_embd),
    };

    for (let i = 0; i < config.n_layer; i++) {
      // ── Grouped Query Attention ──────────────────────────────
      // Q: n_embd → n_head * head_dim
      this.state[`layer${i}.attn_wq`] = matrix(config.n_embd, config.n_embd);
      // K: n_embd → n_kv_head * head_dim
      this.state[`layer${i}.attn_wk`] = matrix(
        config.n_kv_head * config.head_dim,
        config.n_embd,
      );
      // V: n_embd → n_kv_head * head_dim
      this.state[`layer${i}.attn_wv`] = matrix(
        config.n_kv_head * config.head_dim,
        config.n_embd,
      );
      // Output: n_embd → n_embd
      this.state[`layer${i}.attn_wo`] = matrix(config.n_embd, config.n_embd);

      // ── QK-Norm weight vectors (per-head, length = head_dim) ──
      // We store them as 1-row matrices for uniform handling
      this.state[`layer${i}.q_norm`] = matrix(1, config.head_dim);
      this.state[`layer${i}.k_norm`] = matrix(1, config.head_dim);

      if (config.use_moe) {
        // ── MoE FFN ───────────────────────────────────────────
        // Router: n_embd → n_experts  (logits)
        this.state[`layer${i}.router`] = matrix(
          config.n_experts,
          config.n_embd,
        );

        for (let e = 0; e < config.n_experts; e++) {
          this.state[`layer${i}.expert${e}_gate`] = matrix(
            config.ffn_hidden,
            config.n_embd,
          );
          this.state[`layer${i}.expert${e}_up`] = matrix(
            config.ffn_hidden,
            config.n_embd,
          );
          this.state[`layer${i}.expert${e}_down`] = matrix(
            config.n_embd,
            config.ffn_hidden,
          );
        }
      } else {
        // ── Dense SwiGLU FFN ──────────────────────────────────
        this.state[`layer${i}.ffn_gate`] = matrix(
          config.ffn_hidden,
          config.n_embd,
        );
        this.state[`layer${i}.ffn_up`] = matrix(
          config.ffn_hidden,
          config.n_embd,
        );
        this.state[`layer${i}.ffn_down`] = matrix(
          config.n_embd,
          config.ffn_hidden,
        );
      }

      // ── Thinking-mode gate (scalar stored as 1×1 matrix) ─────
      if (config.thinking_mode) {
        this.state[`layer${i}.think_gate`] = matrix(1, 1);
      }
    }

    // ── Flatten all parameters ───────────────────────────────────
    this.params = [];
    Object.values(this.state).forEach((mat) =>
      mat.forEach((row) => row.forEach((p) => this.params.push(p))),
    );

    console.log(
      `Qwen3 micro (${config.use_moe ? "MoE" : "dense"}) — num params:`,
      this.params.length,
    );
  }

  getParams(): Value[] {
    return this.params;
  }

  /**
   * Single-token autoregressive forward pass.
   *
   * @param tokenId  current token index
   * @param posId    position (used for RoPE)
   * @param kvCache  per-layer cache: kvCache[layer] = [{k, v}, ...]
   * @returns        logit vector of shape [vocabSize]
   */
  forward(
    tokenId: number,
    posId: number,
    kvCache: Array<Array<{ k: Value[]; v: Value[] }>>,
  ): Value[] {
    let x = this.state["wte"]![tokenId]!;
    x = rmsnorm(x);

    for (let li = 0; li < this.config.n_layer; li++) {
      // ═══════════════════════════════════════════════════════
      // 1) Grouped Query Attention  +  QK-Norm
      // ═══════════════════════════════════════════════════════
      const residual1 = x;
      x = rmsnorm(x);

      // Projections (no bias — Qwen3 removed QKV bias vs Qwen2)
      const q_raw = linear(x, this.state[`layer${li}.attn_wq`]!);
      const k_raw = linear(x, this.state[`layer${li}.attn_wk`]!);
      const v_raw = linear(x, this.state[`layer${li}.attn_wv`]!);

      // QK-Norm: normalise each head slice before RoPE
      // We share a single norm weight vector across all heads (micro simplification)
      const q_norm_w = this.state[`layer${li}.q_norm`]![0]!;
      const k_norm_w = this.state[`layer${li}.k_norm`]![0]!;

      const applyQKNorm = (vec: Value[], normW: Value[]): Value[] => {
        const normed = qknorm(vec); // unit-norm via RMSNorm
        return normed.map((v, j) => v.mul(normW[j]!)); // learnable scale
      };

      // Split Q into per-query-head slices, apply QK-Norm + RoPE
      const q_heads: Value[][] = [];
      for (let h = 0; h < this.config.n_head; h++) {
        const hs = h * this.config.head_dim;
        const q_h = q_raw.slice(hs, hs + this.config.head_dim);
        const q_normed = applyQKNorm(q_h, q_norm_w);
        q_heads.push(apply_rope(q_normed, posId, this.config.head_dim));
      }

      // Split K/V into per-KV-head slices, apply QK-Norm on K + RoPE
      const k_heads: Value[][] = [];
      const v_heads: Value[][] = [];
      for (let kh = 0; kh < this.config.n_kv_head; kh++) {
        const hs = kh * this.config.head_dim;
        const k_h = k_raw.slice(hs, hs + this.config.head_dim);
        const k_normed = applyQKNorm(k_h, k_norm_w);
        k_heads.push(apply_rope(k_normed, posId, this.config.head_dim));
        v_heads.push(v_raw.slice(hs, hs + this.config.head_dim));
      }

      // Store current position in KV cache
      kvCache[li]!.push({ k: k_heads.flat(), v: v_heads.flat() });

      // GQA attention: each query head group shares one KV head
      const headsPerGroup = this.config.n_head / this.config.n_kv_head;
      let x_attn: Value[] = [];

      for (let h = 0; h < this.config.n_head; h++) {
        const q_h = q_heads[h]!;
        const kvIdx = Math.floor(h / headsPerGroup);

        // Scaled dot-product attention over cache
        const attn_logits: Value[] = [];
        for (let t = 0; t < kvCache[li]!.length; t++) {
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

      x = linear(x_attn, this.state[`layer${li}.attn_wo`]!);
      x = x.map((val, i) => val.add(residual1[i]!));

      // ═══════════════════════════════════════════════════════
      // 2) FFN: dense SwiGLU  OR  sparse MoE (top-k routing)
      // ═══════════════════════════════════════════════════════
      const residual2 = x;
      x = rmsnorm(x);

      let ffn_out: Value[];

      if (this.config.use_moe) {
        // ── Sparse MoE path ──────────────────────────────────
        // Router logits → softmax probs → top-k selection
        const router_logits = linear(x, this.state[`layer${li}.router`]!);
        const router_probs = softmax(router_logits);

        const active = topKIndices(router_probs, this.config.n_active);

        // Renormalise weights over selected experts so they sum to 1
        const activeWeightsRaw = active.map((e) => router_probs[e]!);
        const weightSum = activeWeightsRaw.reduce(
          (acc, w) => acc.add(w),
          new Value(1e-8),
        );
        const activeWeights = activeWeightsRaw.map((w) => w.div(weightSum));

        // Accumulate weighted expert outputs
        ffn_out = Array(this.config.n_embd)
          .fill(0)
          .map(() => new Value(0));

        for (let ai = 0; ai < active.length; ai++) {
          const e = active[ai]!;
          const gate = linear(x, this.state[`layer${li}.expert${e}_gate`]!);
          const up = linear(x, this.state[`layer${li}.expert${e}_up`]!);
          const swiglu = gate.map((g, j) => g.mul(up[j]!.silu()));
          const out = linear(swiglu, this.state[`layer${li}.expert${e}_down`]!);

          for (let j = 0; j < this.config.n_embd; j++) {
            ffn_out[j] = ffn_out[j]!.add(activeWeights[ai]!.mul(out[j]!));
          }
        }
      } else {
        // ── Dense SwiGLU path ────────────────────────────────
        const gate = linear(x, this.state[`layer${li}.ffn_gate`]!);
        const up = linear(x, this.state[`layer${li}.ffn_up`]!);
        const swiglu = gate.map((g, j) => g.mul(up[j]!.silu()));
        ffn_out = linear(swiglu, this.state[`layer${li}.ffn_down`]!);
      }

      // ── Thinking-mode gate (scales FFN contribution) ──────
      if (this.config.thinking_mode) {
        const tg = this.state[`layer${li}.think_gate`]![0]![0]!;
        // sigmoid-gated residual blend: x = residual + sigmoid(tg) * ffn_out
        const gate_val = tg.silu(); // silu ≈ smooth gate (micro proxy)
        ffn_out = ffn_out.map((v) => v.mul(gate_val));
      }

      x = ffn_out.map((val, i) => val.add(residual2[i]!));
    }

    // Final RMSNorm
    x = rmsnorm(x);

    // Separate lm_head projection (Qwen3 does NOT tie embeddings)
    return linear(x, this.state["lm_head"]!);
  }

  async save(path: string = "./qwen3_micro.bin") {
    const rawData: number[] = this.params.map((e) => e.data);
    const floatArray = new Float64Array(rawData);
    await Bun.write(path, floatArray.buffer);
  }

  async load(path: string = "./qwen3_micro.bin") {
    const file = Bun.file(path);
    const bytes = await file.bytes();
    const floatArray = new Float64Array(bytes.buffer);
    this.params.forEach((v, i) => (v.data = floatArray[i]!));
  }
}
