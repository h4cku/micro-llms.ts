/*
The most atomic way to train and inference a GPT in pure,
dependency-free TypeScript (runs with Bun).

Port of @karpathy Python original.
Everything else is just efficiency.
*/

import { GPT, GPTConfig } from "./src/models/gpt";
import { Adam } from "./src/core/optim";
import { Tokenizer } from "./src/core/tokenizer";
import { infere, train } from "./src/utils/train";
import { loadData } from "./src/utils/func";
import { Llama, LlamaConfig } from "./src/models/llama";
import { Deepseek, DeepseekConfig } from "./src/models/deepseek";
import { SmolLM3, SmolLM3Config } from "./src/models/somllm";
import { Qwen3, Qwen3Config } from "./src/models/qwen";

let docs = await loadData();
let tok = new Tokenizer(docs);

let model = undefined;

if (Bun.argv[2] == "gpt") {
  let gptConfig = new GPTConfig();
  model = new GPT(gptConfig, tok);
} else if (Bun.argv[2] == "llama") {
  let llamaConfig = new LlamaConfig();
  model = new Llama(llamaConfig, tok);
} else if (Bun.argv[2] == "deepseek") {
  let deepseekConfig = new DeepseekConfig();
  model = new Deepseek(deepseekConfig, tok);
} else if (Bun.argv[2] == "smollm") {
  let smollmConfig = new SmolLM3Config();
  model = new SmolLM3(smollmConfig, tok);
} else if (Bun.argv[2] == "qwen") {
  let qwenConfig = new Qwen3Config();
  model = new Qwen3(qwenConfig, tok);
}

if (model !== undefined) {
  let optim = new Adam(model.getParams());

  if (Bun.argv[3] == "train") {
    train(model, tok, optim, docs);
    infere(model, tok);
    model.save("models/" + Bun.argv[2] + ".bin");
  } else if (Bun.argv[3] == "infere") {
    await model.load("models/" + Bun.argv[2] + ".bin");
    infere(model, tok);
  }
}
