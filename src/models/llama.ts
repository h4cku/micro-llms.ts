import { matrix } from "../core/matrix";
import { apply_rope, linear, rmsnorm, softmax } from "../core/ops";
import type { Tokenizer } from "../core/tokenizer";
import { Value } from "../core/value";

export class LlamaConfig {
  n_embd: number = 16;
  n_head: number = 4;
  n_kv_head: number = 2;
  n_layer: number = 1;
  block_size: number = 8;
  head_dim = this.n_embd / this.n_head;
  n_rep = this.n_head / this.n_kv_head;
}

export class Llama {
  state: Record<string, Value[][]>;
  params: Value[];
  config: LlamaConfig;

  constructor(config: LlamaConfig, tok: Tokenizer) {
    this.config = config;
    this.state = {
      wte: matrix(tok.vocabSize, config.n_embd),
      lm_head: matrix(tok.vocabSize, config.n_embd),
    };

    for (let i = 0; i < config.n_layer; i++) {
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
      this.state[`layer${i}.ffn_gate`] = matrix(
        4 * config.n_embd,
        config.n_embd,
      );
      this.state[`layer${i}.ffn_up`] = matrix(4 * config.n_embd, config.n_embd);
      this.state[`layer${i}.ffn_down`] = matrix(
        config.n_embd,
        4 * config.n_embd,
      );
    }

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
    keys: Value[][][][],
    values: Value[][][][],
  ): Value[] {
    let x = this.state.wte![tokenId]!;
    x = rmsnorm(x);
    for (let li = 0; li < this.config.n_layer; li++) {
      let x_res = x;
      x = rmsnorm(x);

      const q = linear(x, this.state[`layer${li}.attn_wq`]!);
      const k = linear(x, this.state[`layer${li}.attn_wk`]!);
      const v = linear(x, this.state[`layer${li}.attn_wv`]!);

      const q_heads: Value[][] = [];
      for (let h = 0; h < this.config.n_head; h++) {
        const hs = h * this.config.head_dim;
        const q_h = q.slice(hs, hs + this.config.head_dim);
        q_heads.push(apply_rope(q_h, posId, this.config.head_dim));
      }

      const k_heads: Value[][] = [];
      for (let h = 0; h < this.config.n_kv_head; h++) {
        const hs = h * this.config.head_dim;
        const k_h = k.slice(hs, hs + this.config.head_dim);
        k_heads.push(apply_rope(k_h, posId, this.config.head_dim));
      }

      const v_heads: Value[][] = [];
      for (let h = 0; h < this.config.n_kv_head; h++) {
        const hs = h * this.config.head_dim;
        v_heads.push(v.slice(hs, hs + this.config.head_dim));
      }
      // ====

      keys[li]!.push(k_heads);
      values[li]!.push(v_heads);

      let x_attn: Value[] = [];

      for (let h = 0; h < this.config.n_head; h++) {
        const q_h = q_heads[h];
        const kv_head_idx = Math.floor(h / this.config.n_rep);

        const k_h_all = keys[li]!.map((k_t) => k_t[kv_head_idx]);
        const v_h_all = values[li]!.map((v_t) => v_t[kv_head_idx]);

        // ---- Attention logits ----
        const attn_logits: Value[] = [];

        for (let t = 0; t < k_h_all.length; t++) {
          let dot = new Value(0);

          for (let j = 0; j < this.config.head_dim; j++) {
            dot = dot.add(q_h![j]!.mul(k_h_all![t]![j]!));
          }

          attn_logits.push(dot.div(new Value(Math.sqrt(this.config.head_dim))));
        }

        const attn_weights = softmax(attn_logits);

        // ---- Weighted sum of values ----
        const head_out: Value[] = [];

        for (let j = 0; j < this.config.head_dim; j++) {
          let sum = new Value(0);

          for (let t = 0; t < v_h_all.length; t++) {
            sum = sum.add(attn_weights![t]!.mul(v_h_all![t]![j]!));
          }

          head_out.push(sum);
        }

        x_attn.push(...head_out);
      }

      // Output projection
      x = linear(x_attn, this.state[`layer${li}.attn_wo`]!);

      // Residual
      x = x.map((val, i) => val.add(x_res[i]!));

      // ===== 2) SwiGLU FFN =====
      const ffn_residual = x;
      x = rmsnorm(x);

      const gate = linear(x, this.state[`layer${li}.ffn_gate`]!);
      const up = linear(x, this.state[`layer${li}.ffn_up`]!);

      const swiglu: Value[] = [];

      for (let i = 0; i < gate.length; i++) {
        swiglu.push(gate[i]!.silu().mul(up[i]!));
      }

      x = linear(swiglu, this.state[`layer${li}.ffn_down`]!);

      x = x.map((val, i) => val.add(ffn_residual[i]!));
    }

    // Final norm
    x = rmsnorm(x);

    const logits = linear(x, this.state["lm_head"]!);
    return logits;
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
