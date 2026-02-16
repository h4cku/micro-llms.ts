import type { GPT } from "../models/gpt";
import type { Adam } from "../core/optim";
import type { Tokenizer } from "../core/tokenizer";
import type { Value } from "../core/value";
import { weightedChoice } from "./func";
import { softmax } from "../core/ops";
import type { Llama } from "../models/llama";
import type { Deepseek } from "../models/deepseek";

export function train(
  model: GPT | Llama | Deepseek,
  tok: Tokenizer,
  optim: Adam,
  docs: string[],
  numSteps: number = 500,
) {
  for (let step = 0; step < numSteps; step++) {
    const doc = docs[step % docs.length];
    const tokens = [
      tok.BOS,
      ...Array.from(doc!).map((c) => tok.encode(c)),
      tok.BOS,
    ];
    const n = Math.min(model.config.block_size, tokens.length - 1);

    const keys = Array.from({ length: model.config.n_layer }, () => []);
    const values = Array.from({ length: model.config.n_layer }, () => []);

    const losses: Value[] = [];

    for (let pos = 0; pos < n; pos++) {
      const logits = model.forward(tokens[pos]!, pos, keys, values);
      const probs = softmax(logits);
      losses.push(probs[tokens[pos + 1]!]!.log().neg());
    }

    const loss = losses.reduce((a, b) => a.add(b)).div(n);

    loss.backward();
    optim.step(step, numSteps);
    console.log(`step ${step + 1}/${numSteps} | loss ${loss.data.toFixed(4)}`);
  }
}

export function infere(model: GPT | Llama | Deepseek, tok: Tokenizer) {
  console.log("\n--- inference ---");

  for (let s = 0; s < 20; s++) {
    const keys = Array.from({ length: model.config.n_layer }, () => []);
    const values = Array.from({ length: model.config.n_layer }, () => []);

    let tokenId = tok.BOS;
    let sample = "";

    for (let pos = 0; pos < model.config.block_size; pos++) {
      const logits = model.forward(tokenId, pos, keys, values);
      const probs = softmax(logits.map((l) => l.div(0.5)));
      const idx = weightedChoice(probs.map((p) => p.data));

      if (idx === tok.BOS) break;

      sample += tok.decode(idx);
      tokenId = idx;
    }

    console.log(`sample ${s + 1}: ${sample}`);
  }
}
