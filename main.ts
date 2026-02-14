/*
The most atomic way to train and inference a GPT in pure,
dependency-free TypeScript (runs with Bun).

Port of @karpathy Python original.
Everything else is just efficiency.
*/

import { GPT, GPTConfig } from "./src/gpt";
import { Adam } from "./src/optim";
import { Tokenizer } from "./src/tokenizer";
import { infere, train } from "./src/train";
import { loadData } from "./src/utils";

let docs = await loadData();
let tok = new Tokenizer(docs);

let gptConfig = new GPTConfig();
let gpt = new GPT(gptConfig, tok);
let optim = new Adam(gpt.getParams());

if (Bun.argv[2] == "train") {
  train(gpt, tok, optim, docs);
  infere(gpt, tok);
  gpt.save();
} else if (Bun.argv[2] == "infere") {
  await gpt.load();
  infere(gpt, tok);
}
