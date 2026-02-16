export class Value {
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

  silu(): Value {
// SiLU / Swish: x * sigmoid(x)

  const sig = 1 / (1 + Math.exp(-this.data));
  const outData = this.data * sig;

  // derivative of x * sigmoid(x)
  // = sig * (1 + x * (1 - sig))
  const localGrad = sig * (1 + this.data * (1 - sig));

  return new Value(
    outData,
    [this],
    [localGrad]
  );
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
