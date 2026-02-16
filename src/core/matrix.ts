import { randn } from "../utils/func";
import { Value } from "../core/value";

export function matrix(nout: number, nin: number, std = 0.02) {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(randn(0, std))),
  );
}
