import { Value } from "./value";

export function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((row) =>
    row.reduce((sum, wi, i) => sum.add(wi.mul(x[i]!)), new Value(0)),
  );
}

export function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((a, b) => a.add(b));
  return exps.map((e) => e.div(total));
}

export function rmsnorm(x: Value[]): Value[] {
  const ms = x
    .map((xi) => xi.mul(xi))
    .reduce((a, b) => a.add(b))
    .div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

export function apply_rope(q_or_k: Value[], pos_id: number, head_dim: number){
  const result: Value[] = [];

  for (let i = 0; i < head_dim; i += 2) {
    // Compute rotation angle
    const theta = pos_id / Math.pow(10000, i / head_dim);

    const cosTheta = new Value(Math.cos(theta));
    const sinTheta = new Value(Math.sin(theta));

    // Get the pair
    const x0 = q_or_k[i];
    const x1 = q_or_k[i + 1];

    // Apply rotation:
    // x0' = x0 * cos - x1 * sin
    // x1' = x0 * sin + x1 * cos

    const rotated0 = x0!.mul(cosTheta).sub(x1!.mul(sinTheta));
    const rotated1 = x0!.mul(sinTheta).add(x1!.mul(cosTheta));

    result.push(rotated0);
    result.push(rotated1);
  }

  return result;
}

export function repeat_kv(kv: Value[], n_rep: number): Value[] {
  if (n_rep === 1) {
    return kv;
  }

  const resultLength = kv.length * n_rep;
  const result: Value[] = new Array(resultLength);

  for (let i = 0; i < resultLength; i++) {
    result[i] = kv[Math.floor(i / n_rep)]!;
  }

  return result;
}