import { matrix } from "../core/matrix";
import { linear, rmsnorm, softmax, apply_rope } from "../core/ops";
import type { Tokenizer } from "../core/tokenizer";
import { Value } from "../core/value";

export class DeepseekConfig {
  n_embd: number = 16;
  n_head: number = 4;
  n_layer: number = 1;
  block_size: number = 16;

  // DeepSeek specific
  latent_dim: number = 16; // MLA compression dim
  n_experts: number = 4; // MoE
  n_active: number = 2; // not used in micro version (soft routing)

  head_dim = this.n_embd / this.n_head;
}

export class Deepseek {
  state: Record<string, Value[][]>;
  params: Value[];
  config: DeepseekConfig;

  constructor(config: DeepseekConfig, tok: Tokenizer) {
    this.config = config;

    this.state = {
      wte: matrix(tok.vocabSize, config.n_embd),
      lm_head: matrix(tok.vocabSize, config.n_embd),
    };

    for (let i = 0; i < config.n_layer; i++) {
      // === Multi-Head Latent Attention (MLA) ===
      this.state[`layer${i}.attn_wq`] = matrix(config.n_embd, config.n_embd);

      this.state[`layer${i}.attn_compress_kv`] = matrix(
        config.latent_dim,
        config.n_embd,
      );

      this.state[`layer${i}.attn_wkv`] = matrix(
        config.n_embd,
        config.latent_dim,
      );

      this.state[`layer${i}.attn_wo`] = matrix(config.n_embd, config.n_embd);

      // === Mixture of Experts (MoE) ===
      this.state[`layer${i}.router`] = matrix(config.n_experts, config.n_embd);

      for (let e = 0; e < config.n_experts; e++) {
        this.state[`layer${i}.expert${e}_gate`] = matrix(
          4 * config.n_embd,
          config.n_embd,
        );

        this.state[`layer${i}.expert${e}_up`] = matrix(
          4 * config.n_embd,
          config.n_embd,
        );

        this.state[`layer${i}.expert${e}_down`] = matrix(
          config.n_embd,
          4 * config.n_embd,
        );
      }
    }

    // Flatten params
    this.params = [];
    Object.values(this.state).forEach((mat) =>
      mat.forEach((row) => row.forEach((p) => this.params.push(p))),
    );

    console.log("num params:", this.params.length);
  }

  getParams(): Value[] {
    return this.params;
  }

  forward(
    tokenId: number,
    posId: number,
    keysLatentCache: Value[][][],
  ): Value[] {
    let x = this.state["wte"]![tokenId]!;
    x = rmsnorm(x);

    for (let li = 0; li < this.config.n_layer; li++) {
      // =========================
      // 1) Multi-Head Latent Attention (MLA)
      // =========================

      const residual1 = x;
      x = rmsnorm(x);

      const q = linear(x, this.state[`layer${li}.attn_wq`]!);
      const kv_latent = linear(x, this.state[`layer${li}.attn_compress_kv`]!);

      // Store compressed latent KV
      keysLatentCache[li]!.push(kv_latent);

      // Split Q into heads + RoPE
      const q_heads: Value[][] = [];
      for (let h = 0; h < this.config.n_head; h++) {
        const hs = h * this.config.head_dim;
        const q_h = q.slice(hs, hs + this.config.head_dim);
        q_heads.push(apply_rope(q_h, posId, this.config.head_dim));
      }

      let x_attn: Value[] = [];

      for (let h = 0; h < this.config.n_head; h++) {
        const q_h = q_heads[h]!;

        const attn_logits: Value[] = [];

        for (let t = 0; t < keysLatentCache[li]!.length; t++) {
          const kv_full = linear(
            keysLatentCache[li]![t]!,
            this.state[`layer${li}.attn_wkv`]!,
          );

          const k = kv_full.slice(0, this.config.head_dim);
          const k_rope = apply_rope(k, 0, this.config.head_dim);

          let dot = new Value(0);

          for (let j = 0; j < this.config.head_dim; j++) {
            dot = dot.add(q_h[j]!.mul(k_rope[j]!));
          }

          attn_logits.push(dot.div(new Value(Math.sqrt(this.config.head_dim))));
        }

        const attn_weights = softmax(attn_logits);

        const head_out: Value[] = Array(this.config.head_dim)
          .fill(0)
          .map(() => new Value(0));

        for (let t = 0; t < attn_weights.length; t++) {
          const kv_full = linear(
            keysLatentCache[li]![t]!,
            this.state[`layer${li}.attn_wkv`]!,
          );

          const v = kv_full.slice(
            this.config.head_dim,
            this.config.head_dim * 2,
          );

          for (let j = 0; j < this.config.head_dim; j++) {
            head_out[j] = head_out[j]!.add(attn_weights[t]!.mul(v[j]!));
          }
        }

        x_attn.push(...head_out);
      }

      x = linear(x_attn, this.state[`layer${li}.attn_wo`]!);
      x = x.map((val, i) => val.add(residual1[i]!));

      // =========================
      // 2) Mixture of Experts (MoE)
      // =========================

      const residual2 = x;
      x = rmsnorm(x);

      const router_logits = linear(x, this.state[`layer${li}.router`]!);

      const router_probs = softmax(router_logits);

      const expert_outputs: Value[][] = [];

      for (let e = 0; e < this.config.n_experts; e++) {
        const gate = linear(x, this.state[`layer${li}.expert${e}_gate`]!);

        const up = linear(x, this.state[`layer${li}.expert${e}_up`]!);

        const swiglu: Value[] = [];

        for (let i = 0; i < gate.length; i++) {
          swiglu.push(gate[i]!.silu().mul(up[i]!));
        }

        const out = linear(swiglu, this.state[`layer${li}.expert${e}_down`]!);

        expert_outputs.push(out);
      }

      const output: Value[] = Array(this.config.n_embd)
        .fill(0)
        .map(() => new Value(0));

      for (let e = 0; e < this.config.n_experts; e++) {
        for (let j = 0; j < this.config.n_embd; j++) {
          output[j] = output[j]!.add(
            router_probs[e]!.mul(expert_outputs[e]![j]!),
          );
        }
      }

      x = output.map((val, i) => val.add(residual2[i]!));
    }

    // Final norm
    x = rmsnorm(x);

    return linear(x, this.state["lm_head"]!);
  }

  async save(path: string = "./model.bin") {
    let rawData: number[] = this.params.map((e) => e.data);
    let floatArray = new Float64Array(rawData);
    await Bun.write(path, floatArray.buffer);
  }

  async load(path: string = "./model.bin") {
    const file = Bun.file(path);
    const bytes = await file.bytes();
    let floatArray = new Float64Array(bytes.buffer);
    this.params.forEach((v, i, _) => (v.data = floatArray[i]!));
  }
}
