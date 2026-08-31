// Model ladder. Every entry is a real GPT-style decoder-only transformer:
// RMSNorm, causal multi-head self-attention, GELU MLP (4x), tied input/output
// embeddings, learned via AdamW on next-token cross-entropy.

export interface ModelSpec {
  id: string;
  name: string;
  tier: string;          // human label for parameter class
  d: number;             // model width
  layers: number;
  heads: number;
  ctx: number;           // max context (tokens)
  vocabCap: number;      // tokenizer vocab used by this model
  trainCtx: number;      // default training sequence length
  steps: number;         // default training steps
  lr: number;            // peak learning rate
  memGB: number;         // estimated RAM while training
  heavy?: boolean;       // show a warning
  blurb: string;
}

export const MODELS: ModelSpec[] = [
  {
    id: "nano", name: "Nano", tier: "~86K params",
    d: 32, layers: 3, heads: 2, ctx: 64, vocabCap: 1536, trainCtx: 64,
    steps: 5000, lr: 3e-3, memGB: 0.02,
    blurb: "Trains in about a minute. Learns greetings and short replies fast.",
  },
  {
    id: "micro", name: "Micro", tier: "~0.33M params",
    d: 64, layers: 4, heads: 4, ctx: 96, vocabCap: 2048, trainCtx: 96,
    steps: 4000, lr: 2e-3, memGB: 0.05,
    blurb: "A small step up. Picks up multi-turn structure and simple facts.",
  },
  {
    id: "small", name: "Small", tier: "~1.6M params",
    d: 128, layers: 6, heads: 4, ctx: 128, vocabCap: 3072, trainCtx: 128,
    steps: 3000, lr: 1.2e-3, memGB: 0.15,
    blurb: "The default. Good balance of speed and coherence for chatting.",
  },
  {
    id: "medium", name: "Medium", tier: "~7.3M params",
    d: 256, layers: 8, heads: 8, ctx: 128, vocabCap: 4096, trainCtx: 128,
    steps: 1200, lr: 7e-4, memGB: 0.6,
    blurb: "Noticeably steadier grammar. A few minutes per run on a laptop.",
  },
  {
    id: "big", name: "Big", tier: "~13.6M params",
    d: 320, layers: 10, heads: 5, ctx: 128, vocabCap: 4096, trainCtx: 96,
    steps: 500, lr: 5e-4, memGB: 1.1,
    blurb: "Wider and deeper. Give it time and a quiet tab.",
  },
  {
    id: "massive", name: "Massive", tier: "~42M params",
    d: 512, layers: 12, heads: 8, ctx: 96, vocabCap: 8192, trainCtx: 64,
    steps: 120, lr: 3e-4, memGB: 1.9, heavy: true,
    blurb: "Serious hardware territory. Expect seconds per step, not milliseconds.",
  },
  {
    id: "giga", name: "Giga", tier: "~107M params",
    d: 832, layers: 12, heads: 13, ctx: 80, vocabCap: 8192, trainCtx: 48,
    steps: 30, lr: 2e-4, memGB: 3.4, heavy: true,
    blurb: "Included for completeness. Needs a big desktop and a lot of patience.",
  },
];

// exact parameter count: tied embedding V*d + L*(12d^2 + 2d) + d
export function paramCount(spec: ModelSpec, vocab: number): number {
  const d = spec.d;
  return vocab * d + spec.layers * (12 * d * d + 2 * d) + d;
}

export function formatParams(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return String(n);
}

// rough training memory: fp32 weights + grads + 2x Adam moments = 16 B/param
export function memoryEstimate(spec: ModelSpec, vocab: number): number {
  const p = paramCount(spec, vocab);
  const weights = p * 16;
  const acts = spec.trainCtx * spec.d * spec.layers * 48;
  return weights + acts;
}
