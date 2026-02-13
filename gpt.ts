/*
The most atomic way to train and inference a GPT in pure,
dependency-free TypeScript (runs with Bun).

Port of @karpathy Python original.
Everything else is just efficiency.
*/

///////////////////////////////
// Utilities
///////////////////////////////

function randn(mean = 0, std = 1): number {
  // Box–Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function shuffle<T>(arr: T[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
}

function weightedChoice(weights: number[]): number {
  const sum = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i]!;
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

///////////////////////////////
// Autograd
///////////////////////////////

class Value {
  data: number;
  grad: number = 0;
  private _children: Value[];
  private _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this._children = children;
    this._localGrads = localGrads;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(exp: number): Value {
    return new Value(this.data ** exp, [this], [exp * this.data ** (exp - 1)]);
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }

  relu(): Value {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    return this.add(other instanceof Value ? other.neg() : -other);
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  backward() {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const build = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        v._children.forEach(build);
        topo.push(v);
      }
    };

    build(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      v!._children.forEach((child, j) => {
        child.grad += v!._localGrads[j]! * v!.grad;
      });
    }
  }
}

///////////////////////////////
// Load dataset
///////////////////////////////

let text = await Bun.file("input.txt").text().catch(async () => {
  const url =
    "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";
  const res = await fetch(url);
  const t = await res.text();
  await Bun.write("input.txt", t);
  return t;
});

let docs = text
  .trim()
  .split("\n")
  .map((l) => l.trim())
  .filter(Boolean);

shuffle(docs);
console.log("num docs:", docs.length);

///////////////////////////////
// Tokenizer
///////////////////////////////

const uchars = Array.from(new Set(docs.join(""))).sort();
const BOS = uchars.length;
const vocabSize = uchars.length + 1;
console.log("vocab size:", vocabSize);

///////////////////////////////
// Model hyperparameters
///////////////////////////////

const n_embd = 16;
const n_head = 4;
const n_layer = 1;
const block_size = 8;
const head_dim = n_embd / n_head;

function matrix(nout: number, nin: number, std = 0.02) {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(randn(0, std)))
  );
}

///////////////////////////////
// Parameters
///////////////////////////////

const state: Record<string, Value[][]> = {
  wte: matrix(vocabSize, n_embd),
  wpe: matrix(block_size, n_embd),
  lm_head: matrix(vocabSize, n_embd),
};

for (let i = 0; i < n_layer; i++) {
  state[`layer${i}.attn_wq`] = matrix(n_embd, n_embd);
  state[`layer${i}.attn_wk`] = matrix(n_embd, n_embd);
  state[`layer${i}.attn_wv`] = matrix(n_embd, n_embd);
  state[`layer${i}.attn_wo`] = matrix(n_embd, n_embd, 0);
  state[`layer${i}.mlp_fc1`] = matrix(4 * n_embd, n_embd);
  state[`layer${i}.mlp_fc2`] = matrix(n_embd, 4 * n_embd, 0);
}

const params: Value[] = [];
Object.values(state).forEach((mat) =>
  mat.forEach((row) => row.forEach((p) => params.push(p)))
);

console.log("num params:", params.length);

///////////////////////////////
// Model ops
///////////////////////////////

function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((row) =>
    row.reduce((sum, wi, i) => sum.add(wi.mul(x[i]!)), new Value(0))
  );
}

function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((a, b) => a.add(b));
  return exps.map((e) => e.div(total));
}

function rmsnorm(x: Value[]): Value[] {
  const ms = x
    .map((xi) => xi.mul(xi))
    .reduce((a, b) => a.add(b))
    .div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

function gpt(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][]
): Value[] {
  let x = state.wte![tokenId]!.map((t, i) =>
    t.add(state.wpe![posId]![i]!)
  );
  x = rmsnorm(x);

  for (let li = 0; li < n_layer; li++) {
    let x_res = x;
    x = rmsnorm(x);

    const q = linear(x, state[`layer${li}.attn_wq`]!);
    const k = linear(x, state[`layer${li}.attn_wk`]!);
    const v = linear(x, state[`layer${li}.attn_wv`]!);

    keys[li]!.push(k);
    values[li]!.push(v);

    let x_attn: Value[] = [];

    for (let h = 0; h < n_head; h++) {
      const hs = h * head_dim;
      const qh = q.slice(hs, hs + head_dim);

      const kh = keys[li]!.map((kk) => kk.slice(hs, hs + head_dim));
      const vh = values[li]!.map((vv) => vv.slice(hs, hs + head_dim));

      const logits = kh.map((kt) =>
        qh
          .map((qj, j) => qj.mul(kt[j]!))
          .reduce((a, b) => a.add(b))
          .div(Math.sqrt(head_dim))
      );

      const weights = softmax(logits);

      for (let j = 0; j < head_dim; j++) {
        const val = weights
          .map((wt, t) => wt.mul(vh[t]![j]!))
          .reduce((a, b) => a.add(b));
        x_attn.push(val);
      }
    }

    x = linear(x_attn, state[`layer${li}.attn_wo`]!);
    x = x.map((xi, i) => xi.add(x_res[i]!));

    x_res = x;
    x = rmsnorm(x);
    x = linear(x, state[`layer${li}.mlp_fc1`]!);
    x = x.map((xi) => xi.relu().pow(2));
    x = linear(x, state[`layer${li}.mlp_fc2`]!);
    x = x.map((xi, i) => xi.add(x_res[i]!));
  }

  return linear(x, state.lm_head!);
}

///////////////////////////////
// Training (Adam)
///////////////////////////////

const lr = 1e-2;
const beta1 = 0.9;
const beta2 = 0.95;
const eps = 1e-8;

const m = Array(params.length).fill(0);
const v = Array(params.length).fill(0);

const numSteps = 500;

for (let step = 0; step < numSteps; step++) {
  const doc = docs[step % docs.length];
  const tokens = [
    BOS,
    ...Array.from(doc!).map((c) => uchars.indexOf(c)),
    BOS,
  ];
  const n = Math.min(block_size, tokens.length - 1);

  const keys = Array.from({ length: n_layer }, () => []);
  const values = Array.from({ length: n_layer }, () => []);

  const losses: Value[] = [];

  for (let pos = 0; pos < n; pos++) {
    const logits = gpt(tokens[pos]!, pos, keys, values);
    const probs = softmax(logits);
    losses.push(probs[tokens[pos + 1]!]!.log().neg());
  }

  const loss = losses
    .reduce((a, b) => a.add(b))
    .div(n);

  loss.backward();

  const lr_t =
    lr * 0.5 * (1 + Math.cos(Math.PI * step / numSteps));

  params.forEach((p, i) => {
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2;
    const mhat = m[i] / (1 - beta1 ** (step + 1));
    const vhat = v[i] / (1 - beta2 ** (step + 1));
    p.data -= lr_t * mhat / (Math.sqrt(vhat) + eps);
    p.grad = 0;
  });

  console.log(`step ${step + 1}/${numSteps} | loss ${loss.data.toFixed(4)}`);
}

///////////////////////////////
// Inference
///////////////////////////////

console.log("\n--- inference ---");

for (let s = 0; s < 20; s++) {
  const keys = Array.from({ length: n_layer }, () => []);
  const values = Array.from({ length: n_layer }, () => []);

  let tokenId = BOS;
  let sample = "";

  for (let pos = 0; pos < block_size; pos++) {
    const logits = gpt(tokenId, pos, keys, values);
    const probs = softmax(logits.map((l) => l.div(0.5)));
    const idx = weightedChoice(probs.map((p) => p.data));

    if (idx === BOS) break;

    sample += uchars[idx];
    tokenId = idx;
  }

  console.log(`sample ${s + 1}: ${sample}`);
}

