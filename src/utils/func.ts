export function randn(mean = 0, std = 1): number {
  // Box–Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function shuffle<T>(arr: T[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j]!, arr[i]!];
  }
}

export function weightedChoice(weights: number[]): number {
  const sum = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i]!;
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

export async function loadData(): Promise<string[]> {
  let text = await Bun.file("input.txt")
    .text()
    .catch(async () => {
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
  return docs;
}
