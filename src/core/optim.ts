import type { Value } from "./value";

export class Adam {
  params: Value[];
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  m: any[];
  v: any[];

  constructor(
    params: Value[],
    lr: number = 1e-2,
    beta1: number = 0.9,
    beta2: number = 0.95,
    eps: number = 1e-8,
  ) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.m = Array(this.params.length).fill(0);
    this.v = Array(this.params.length).fill(0);
  }

  step(step: number, numSteps: number) {
    const lr_t = this.lr * 0.5 * (1 + Math.cos((Math.PI * step) / numSteps));

    this.params.forEach((p, i) => {
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * p.grad;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * p.grad ** 2;
      const mhat = this.m[i] / (1 - this.beta1 ** (step + 1));
      const vhat = this.v[i] / (1 - this.beta2 ** (step + 1));
      p.data -= (lr_t * mhat) / (Math.sqrt(vhat) + this.eps);
      p.grad = 0;
    });
  }
}
