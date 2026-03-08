import { matrix } from "../core/matrix";
import type { Tokenizer } from "../core/tokenizer";
import { Value } from "../core/value";
import { softmax, linear, rmsnorm } from "../core/ops";

export class GPTConfig {
  n_embd: number = 16;
  n_head: number = 4;
  n_layer: number = 1;
  block_size: number = 8;
  head_dim = this.n_embd / this.n_head;
}

/**
 * Micro GPT — a toy implementation capturing the key architectural ideas
 * of a GPT-like decoder-only transformer model.
 *
 * Key architecture highlights:
 *
 *  1. Decoder-Only Transformer Blocks
 *     • Each layer consists of a multi-head self-attention mechanism followed by a two-layer Feed-Forward Network (FFN).
 *
 *  2. Multi-Head Self-Attention
 *     • Queries (Q), Keys (K), and Values (V) are projected from the input.
 *     • Attention logits are calculated as a dot product of Q and K, scaled by the square root of the head dimension.
 *     • `softmax` is applied to the logits to obtain attention weights.
 *     • The output of each head is a weighted sum of V.
 *     • Outputs from all heads are concatenated and then linearly projected.
 *
 *  3. Positional Embeddings
 *     • Uses both Word Token Embeddings (`wte`) and Word Positional Embeddings (`wpe`).
 *     • The input to the transformer blocks is the sum of these two embeddings.
 *
 *  4. Feed-Forward Network (FFN)
 *     • A two-layer MLP (`mlp_fc1` and `mlp_fc2`) with a non-linear activation function (ReLU squared in this micro-implementation: `relu().pow(2)`).
 *
 *  5. RMSNorm
 *     • Applied at the beginning of each transformer block (before attention and FFN) and after the input embedding.
 *
 *  6. Residual Connections
 *     • Applied after the attention block and after the FFN block.
 *
 *  7. Separate Embeddings
 *     • `wte`, `wpe`, and `lm_head` are distinct projections.
 *
 * Micro defaults are intentionally tiny for autograd / educational use.
 */
export class GPT {
  state: Record<string, Value[][]>;
  params: Value[];
  config: GPTConfig;

  constructor(config: GPTConfig, tok: Tokenizer) {
    this.config = config;
    this.state = {
      wte: matrix(tok.vocabSize, config.n_embd),
      wpe: matrix(config.block_size, config.n_embd),
      lm_head: matrix(tok.vocabSize, config.n_embd),
    };

    for (let i = 0; i < config.n_layer; i++) {
      this.state[`layer${i}.attn_wq`] = matrix(config.n_embd, config.n_embd);
      this.state[`layer${i}.attn_wk`] = matrix(config.n_embd, config.n_embd);
      this.state[`layer${i}.attn_wv`] = matrix(config.n_embd, config.n_embd);
      this.state[`layer${i}.attn_wo`] = matrix(config.n_embd, config.n_embd, 0);
      this.state[`layer${i}.mlp_fc1`] = matrix(
        4 * config.n_embd,
        config.n_embd,
      );
      this.state[`layer${i}.mlp_fc2`] = matrix(
        config.n_embd,
        4 * config.n_embd,
        0,
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
    keys: Value[][][],
    values: Value[][][],
  ): Value[] {
    let x = this.state.wte![tokenId]!.map((t, i) =>
      t.add(this.state.wpe![posId]![i]!),
    );
    x = rmsnorm(x);

    for (let li = 0; li < this.config.n_layer; li++) {
      let x_res = x;
      x = rmsnorm(x);

      const q = linear(x, this.state[`layer${li}.attn_wq`]!);
      const k = linear(x, this.state[`layer${li}.attn_wk`]!);
      const v = linear(x, this.state[`layer${li}.attn_wv`]!);

      keys[li]!.push(k);
      values[li]!.push(v);

      let x_attn: Value[] = [];

      for (let h = 0; h < this.config.n_head; h++) {
        const hs = h * this.config.head_dim;
        const qh = q.slice(hs, hs + this.config.head_dim);

        const kh = keys[li]!.map((kk) =>
          kk.slice(hs, hs + this.config.head_dim),
        );
        const vh = values[li]!.map((vv) =>
          vv.slice(hs, hs + this.config.head_dim),
        );

        const logits = kh.map((kt) =>
          qh
            .map((qj, j) => qj.mul(kt[j]!))
            .reduce((a, b) => a.add(b))
            .div(Math.sqrt(this.config.head_dim)),
        );

        const weights = softmax(logits);

        for (let j = 0; j < this.config.head_dim; j++) {
          const val = weights
            .map((wt, t) => wt.mul(vh[t]![j]!))
            .reduce((a, b) => a.add(b));
          x_attn.push(val);
        }
      }

      x = linear(x_attn, this.state[`layer${li}.attn_wo`]!);
      x = x.map((xi, i) => xi.add(x_res[i]!));

      x_res = x;
      x = rmsnorm(x);
      x = linear(x, this.state[`layer${li}.mlp_fc1`]!);
      x = x.map((xi) => xi.relu().pow(2));
      x = linear(x, this.state[`layer${li}.mlp_fc2`]!);
      x = x.map((xi, i) => xi.add(x_res[i]!));
    }

    return linear(x, this.state.lm_head!);
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
