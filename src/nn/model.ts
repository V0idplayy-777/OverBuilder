// The transformer. A small but real decoder-only GPT:
// tied embedding / output head, pre-RMSNorm blocks, causal multi-head
// attention, GELU mlp (4x). Trained with AdamW on next-token cross-entropy.
//
// Backprop is a tiny tape engine. Every heavy GEMM routes through the
// (WASM-backed) kernel layer.

import { matNT, matNN, matVec, transpose } from "./kernels";
import type { ModelSpec } from "./configs";

// ---------------------------------------------------------------- utils

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeRandn(seed: number) {
  const rng = mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = rng();
    v = rng();
    const m = Math.sqrt(-2 * Math.log(u));
    spare = m * Math.sin(2 * Math.PI * v);
    return m * Math.cos(2 * Math.PI * v);
  };
}

export interface Param {
  name: string;
  d: Float32Array;
  g: Float32Array;
  m: Float32Array;
  v: Float32Array;
  wd: boolean;
}

export interface Tensor { d: Float32Array; g: Float32Array | null; }
type Tape = Array<() => void>;

const need = (t: Tensor, fill = true): Float32Array => {
  if (!t.g) {
    t.g = new Float32Array(t.d.length);
    if (!fill) return t.g;
  }
  return t.g;
};

// ---------------------------------------------------------------- ops

function linear(tape: Tape, x: Tensor, w: Param, T: number, outDim: number, inDim: number): Tensor {
  const out: Tensor = { d: new Float32Array(T * outDim), g: null };
  matNT(x.d, w.d, out.d, T, outDim, inDim, 0); // y = x w^T
  tape.push(() => {
    const dy = out.g!;
    const dyT = new Float32Array(dy.length);
    transpose(dy, dyT, T, outDim);
    matNN(dyT, x.d, w.g, outDim, inDim, T, 1);  // dW += dy^T x
    matNN(dy, w.d, need(x), T, inDim, outDim, 1); // dX += dy W
  });
  return out;
}

function rmsnorm(tape: Tape, x: Tensor, w: Param, T: number, d: number): Tensor {
  const out: Tensor = { d: new Float32Array(T * d), g: null };
  const invRms = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let ss = 0;
    for (let i = 0; i < d; i++) ss += x.d[o + i] * x.d[o + i];
    const s = 1 / Math.sqrt(ss / d + 1e-5);
    invRms[t] = s;
    for (let i = 0; i < d; i++) out.d[o + i] = x.d[o + i] * s * w.d[i];
  }
  tape.push(() => {
    const dy = out.g!, dx = need(x);
    for (let t = 0; t < T; t++) {
      const o = t * d;
      const s = invRms[t];
      let dotp = 0;
      for (let i = 0; i < d; i++) {
        dotp += dy[o + i] * w.d[i] * x.d[o + i];
        w.g[i] += dy[o + i] * x.d[o + i] * s;
      }
      const c = (s * s) / d * dotp;
      for (let i = 0; i < d; i++) dx[o + i] += dy[o + i] * w.d[i] * s - x.d[o + i] * c;
    }
  });
  return out;
}

function gelu(tape: Tape, x: Tensor): Tensor {
  const n = x.d.length;
  const out: Tensor = { d: new Float32Array(n), g: null };
  const C = 0.7978845608028654;
  for (let i = 0; i < n; i++) {
    const v = x.d[i];
    const u = C * (v + 0.044715 * v * v * v);
    out.d[i] = 0.5 * v * (1 + Math.tanh(u));
  }
  tape.push(() => {
    const dy = out.g!, dx = need(x);
    for (let i = 0; i < n; i++) {
      const v = x.d[i];
      const u = C * (v + 0.044715 * v * v * v);
      const th = Math.tanh(u);
      const du = C * (1 + 3 * 0.044715 * v * v);
      dx[i] += dy[i] * 0.5 * (1 + th + v * (1 - th * th) * du);
    }
  });
  return out;
}

function add(tape: Tape, a: Tensor, b: Tensor): Tensor {
  const n = a.d.length;
  const out: Tensor = { d: new Float32Array(n), g: null };
  for (let i = 0; i < n; i++) out.d[i] = a.d[i] + b.d[i];
  tape.push(() => {
    const g = out.g!, ga = need(a), gb = need(b);
    for (let i = 0; i < n; i++) { ga[i] += g[i]; gb[i] += g[i]; }
  });
  return out;
}

// causal multi-head attention over the whole sequence
function attention(tape: Tape, x: Tensor, p: LayerParams, T: number, d: number, H: number): Tensor {
  const dh = d / H;
  const scale = 1 / Math.sqrt(dh);
  const qkv = linear(tape, x, p.qkv, T, 3 * d, d);
  const merged: Tensor = { d: new Float32Array(T * d), g: null };
  const backSlices: Array<() => void> = [];

  for (let h = 0; h < H; h++) {
    const q = new Float32Array(T * dh), k = new Float32Array(T * dh), v = new Float32Array(T * dh);
    for (let t = 0; t < T; t++) {
      const src = t * 3 * d + h * dh, dst = t * dh;
      for (let i = 0; i < dh; i++) {
        q[dst + i] = qkv.d[src + i];
        k[dst + i] = qkv.d[d + src + i];
        v[dst + i] = qkv.d[2 * d + src + i];
      }
    }
    // scores + causal mask + softmax
    const probs = new Float32Array(T * T);
    matNT(q, k, probs, T, T, dh, 0);
    for (let t = 0; t < T; t++) {
      const o = t * T;
      let mx = -Infinity;
      for (let s2 = 0; s2 <= t; s2++) { probs[o + s2] *= scale; if (probs[o + s2] > mx) mx = probs[o + s2]; }
      let sm = 0;
      for (let s2 = 0; s2 <= t; s2++) { const e = Math.exp(probs[o + s2] - mx); probs[o + s2] = e; sm += e; }
      for (let s2 = 0; s2 <= t; s2++) probs[o + s2] /= sm;
      for (let s2 = t + 1; s2 < T; s2++) probs[o + s2] = 0;
    }
    // context
    const ctx = new Float32Array(T * dh);
    matNN(probs, v, ctx, T, dh, T, 0);
    for (let t = 0; t < T; t++) {
      const dst = t * d + h * dh, src = t * dh;
      for (let i = 0; i < dh; i++) merged.d[dst + i] = ctx[src + i];
    }

    backSlices.push(() => {
      const dctx = new Float32Array(T * dh);
      const mg = merged.g!;
      for (let t = 0; t < T; t++) {
        const src = t * d + h * dh, dst = t * dh;
        for (let i = 0; i < dh; i++) dctx[dst + i] = mg[src + i];
      }
      // dP = dctx v^T ; dV = P^T dctx
      const dprobs = new Float32Array(T * T);
      matNT(dctx, v, dprobs, T, T, dh, 0);
      const pT = new Float32Array(T * T);
      transpose(probs, pT, T, T);
      const dv = new Float32Array(T * dh);
      matNN(pT, dctx, dv, T, dh, T, 0);
      // softmax backward
      const ds = new Float32Array(T * T);
      for (let t = 0; t < T; t++) {
        const o = t * T;
        let dotp = 0;
        for (let s2 = 0; s2 <= t; s2++) dotp += dprobs[o + s2] * probs[o + s2];
        for (let s2 = 0; s2 <= t; s2++) ds[o + s2] = probs[o + s2] * (dprobs[o + s2] - dotp) * scale;
      }
      // dQ = ds k ; dK = ds^T q
      const dq = new Float32Array(T * dh), dk = new Float32Array(T * dh);
      matNN(ds, k, dq, T, dh, T, 0);
      const dsT = new Float32Array(T * T);
      transpose(ds, dsT, T, T);
      matNN(dsT, q, dk, T, dh, T, 0);
      // scatter into dqkv
      const gq = need(qkv);
      for (let t = 0; t < T; t++) {
        const dst = t * 3 * d + h * dh, src = t * dh;
        for (let i = 0; i < dh; i++) {
          gq[dst + i] += dq[src + i];
          gq[d + dst + i] += dk[src + i];
          gq[2 * d + dst + i] += dv[src + i];
        }
      }
    });
  }
  tape.push(() => { for (let h = H - 1; h >= 0; h--) backSlices[h](); });
  return linear(tape, merged, p.proj, T, d, d);
}

// ---------------------------------------------------------------- model

interface LayerParams {
  qkv: Param; proj: Param; f1: Param; f2: Param; r1: Param; r2: Param;
}

export class GPT {
  spec: ModelSpec;
  d: number; L: number; H: number; V: number;
  emb: Param;
  layers: LayerParams[] = [];
  rf: Param;
  params: Param[] = [];
  stepCount = 0;

  constructor(spec: ModelSpec, vocab: number, seed = 1234) {
    this.spec = spec;
    this.V = vocab;
    const d = spec.d, L = spec.layers, H = spec.heads;
    this.d = d; this.L = L; this.H = H;
    const randn = makeRandn(seed);
    const mk = (name: string, n: number, std: number, wd: boolean): Param => {
      const p: Param = {
        name,
        d: new Float32Array(n), g: new Float32Array(n),
        m: new Float32Array(n), v: new Float32Array(n), wd,
      };
      if (std > 0) for (let i = 0; i < n; i++) p.d[i] = randn() * std;
      else p.d.fill(1); // norm gains
      this.params.push(p);
      return p;
    };
    this.emb = mk("emb", vocab * d, 0.02, true);
    const resScale = 0.02 / Math.sqrt(2 * L);
    for (let l = 0; l < L; l++) {
      this.layers.push({
        r1: mk(`l${l}.r1`, d, 0, false),
        qkv: mk(`l${l}.qkv`, 3 * d * d, 0.02, true),
        proj: mk(`l${l}.proj`, d * d, resScale, true),
        r2: mk(`l${l}.r2`, d, 0, false),
        f1: mk(`l${l}.f1`, 4 * d * d, 0.02, true),
        f2: mk(`l${l}.f2`, 4 * d * d, resScale, true),
      });
    }
    this.rf = mk("rf", d, 0, false);
  }

  zeroGrads() {
    for (const p of this.params) p.g.fill(0);
  }

  // forward + backward for one window of ids (length T+1), returns mean loss
  trainStep(ids: Uint32Array | number[], T: number): number {
    const d = this.d;
    const tape: Tape = [];
    // embedding gather
    const x: Tensor = { d: new Float32Array(T * d), g: null };
    for (let t = 0; t < T; t++) {
      const id = ids[t];
      const src = id * d, dst = t * d;
      for (let i = 0; i < d; i++) x.d[dst + i] = this.emb.d[src + i];
    }
    tape.push(() => {
      const gx = x.g!;
      for (let t = 0; t < T; t++) {
        const id = ids[t];
        const dst = id * d, src = t * d;
        for (let i = 0; i < d; i++) this.emb.g[dst + i] += gx[src + i];
      }
    });

    let h = x;
    for (const p of this.layers) {
      const n1 = rmsnorm(tape, h, p.r1, T, d);
      h = add(tape, h, attention(tape, n1, p, T, d, this.H));
      const n2 = rmsnorm(tape, h, p.r2, T, d);
      h = add(tape, h, linear(tape, gelu(tape, linear(tape, n2, p.f1, T, 4 * d, d)), p.f2, T, d, 4 * d));
    }
    const hf = rmsnorm(tape, h, this.rf, T, d);
    const logits = linear(tape, hf, this.emb, T, this.V, d); // tied head

    // fused softmax + cross-entropy, writing d(logits)
    const lg = need(logits, false)!;
    let loss = 0;
    for (let t = 0; t < T; t++) {
      const o = t * this.V;
      const tgt = ids[t + 1];
      let mx = -Infinity;
      for (let i = 0; i < this.V; i++) if (logits.d[o + i] > mx) mx = logits.d[o + i];
      let sm = 0;
      for (let i = 0; i < this.V; i++) { lg[o + i] = Math.exp(logits.d[o + i] - mx); sm += lg[o + i]; }
      loss += Math.log(sm) - (logits.d[o + tgt] - mx);
      const inv = 1 / (sm * T);
      for (let i = 0; i < this.V; i++) lg[o + i] *= inv;
      lg[o + tgt] -= 1 / T;
    }
    loss /= T;

    for (let i = tape.length - 1; i >= 0; i--) tape[i]();
    return loss;
  }

  clipGrads(maxNorm: number): number {
    let ss = 0;
    for (const p of this.params) {
      const g = p.g;
      for (let i = 0; i < g.length; i++) ss += g[i] * g[i];
    }
    const norm = Math.sqrt(ss);
    if (norm > maxNorm && norm > 0) {
      const s = maxNorm / norm;
      for (const p of this.params) {
        const g = p.g;
        for (let i = 0; i < g.length; i++) g[i] *= s;
      }
    }
    return norm;
  }

  adamw(lr: number, wd: number, b1 = 0.9, b2 = 0.98, eps = 1e-8) {
    this.stepCount++;
    const bc1 = 1 - Math.pow(b1, this.stepCount);
    const bc2 = 1 - Math.pow(b2, this.stepCount);
    for (const p of this.params) {
      const { d: w, g, m, v } = p;
      const decay = p.wd ? 1 - lr * wd : 1;
      const c1 = 1 - b1, c2 = 1 - b2;
      const k = lr / bc1;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i];
        m[i] = b1 * m[i] + c1 * gi;
        v[i] = b2 * v[i] + c2 * gi * gi;
        w[i] = w[i] * decay - (k * m[i]) / (Math.sqrt(v[i] / bc2) + eps);
      }
    }
  }

  // ------------------------------------------------------------ inference

  createCache() {
    const sz = this.spec.ctx * this.d;
    return {
      pos: 0,
      k: this.layers.map(() => new Float32Array(sz)),
      v: this.layers.map(() => new Float32Array(sz)),
    };
  }

  // advance one token through the network using the kv cache; returns logits
  stepToken(id: number, cache: { pos: number; k: Float32Array[]; v: Float32Array[] }): Float32Array {
    const d = this.d, H = this.H, dh = d / H;
    const scale = 1 / Math.sqrt(dh);
    const x = new Float32Array(d);
    x.set(this.emb.d.subarray(id * d, (id + 1) * d));
    const pos = cache.pos;

    const qkv = new Float32Array(3 * d);
    const attn = new Float32Array(d);
    const tmp = new Float32Array(d);
    const n1 = new Float32Array(d);
    const ff = new Float32Array(4 * d);
    const ff2 = new Float32Array(d);

    const rms = (src: Float32Array, g: Float32Array, dst: Float32Array) => {
      let ss = 0;
      for (let i = 0; i < d; i++) ss += src[i] * src[i];
      const s = 1 / Math.sqrt(ss / d + 1e-5);
      for (let i = 0; i < d; i++) dst[i] = src[i] * s * g[i];
    };

    for (let l = 0; l < this.L; l++) {
      const p = this.layers[l];
      rms(x, p.r1.d, n1);
      matVec(n1, p.qkv.d, qkv, 3 * d, d);
      cache.k[l].set(qkv.subarray(d, 2 * d), pos * d);
      cache.v[l].set(qkv.subarray(2 * d, 3 * d), pos * d);
      const kc = cache.k[l], vc = cache.v[l];
      for (let h = 0; h < H; h++) {
        const qo = h * dh;
        // scores
        const sc = new Float32Array(pos + 1);
        let mx = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          const ko = t * d + qo;
          for (let i = 0; i < dh; i++) s += qkv[qo + i] * kc[ko + i];
          sc[t] = s * scale;
          if (sc[t] > mx) mx = sc[t];
        }
        let sm = 0;
        for (let t = 0; t <= pos; t++) { sc[t] = Math.exp(sc[t] - mx); sm += sc[t]; }
        for (let i = 0; i < dh; i++) attn[qo + i] = 0;
        for (let t = 0; t <= pos; t++) {
          const w = sc[t] / sm;
          const vo = t * d + qo;
          for (let i = 0; i < dh; i++) attn[qo + i] += w * vc[vo + i];
        }
      }
      matVec(attn, p.proj.d, tmp, d, d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
      rms(x, p.r2.d, n1);
      matVec(n1, p.f1.d, ff, 4 * d, d);
      const C = 0.7978845608028654;
      for (let i = 0; i < 4 * d; i++) {
        const v = ff[i];
        ff[i] = 0.5 * v * (1 + Math.tanh(C * (v + 0.044715 * v * v * v)));
      }
      matVec(ff, p.f2.d, ff2, d, 4 * d);
      for (let i = 0; i < d; i++) x[i] += ff2[i];
    }
    rms(x, this.rf.d, n1);
    const logits = new Float32Array(this.V);
    matVec(n1, this.emb.d, logits, this.V, d);
    cache.pos++;
    return logits;
  }

  packedWeights(): Float32Array {
    const total = this.params.reduce((s, p) => s + p.d.length, 0);
    const out = new Float32Array(total);
    let o = 0;
    for (const p of this.params) { out.set(p.d, o); o += p.d.length; }
    return out;
  }

  loadPacked(weights: Float32Array) {
    let o = 0;
    for (const p of this.params) {
      p.d.set(weights.subarray(o, o + p.d.length));
      o += p.d.length;
      p.m.fill(0); p.v.fill(0);
    }
    this.stepCount = 0;
  }
}

// sampling with temperature, top-k, top-p, repetition penalty
export function sampleNext(
  logits: Float32Array,
  seen: number[] | Set<number>,
  temp: number,
  topK: number,
  topP: number,
  repPenalty: number,
): number {
  const n = logits.length;
  const work = Float32Array.from(logits);
  if (repPenalty !== 1) {
    for (const id of seen) {
      if (id < n) work[id] = work[id] > 0 ? work[id] / repPenalty : work[id] * repPenalty;
    }
  }
  let mx = -Infinity;
  for (let i = 0; i < n; i++) { work[i] /= temp; if (work[i] > mx) mx = work[i]; }
  let sm = 0;
  for (let i = 0; i < n; i++) { work[i] = Math.exp(work[i] - mx); sm += work[i]; }
  for (let i = 0; i < n; i++) work[i] /= sm;

  const order = [...work.keys()].sort((a, b) => work[b] - work[a]);
  const kN = Math.max(1, Math.min(topK, n));
  let chosen = order[n - 1];
  let cum = 0;
  for (let i = 0; i < kN; i++) {
    const id = order[i];
    cum += work[id];
    if (cum >= topP || i === kN - 1) {
      // sample among [0..i] renormalized
      let r = Math.random() * Math.min(cum, topP);
      let acc = 0;
      chosen = id;
      for (let j = 0; j <= i; j++) {
        acc += work[order[j]];
        if (acc >= r) { chosen = order[j]; break; }
      }
      break;
    }
  }
  return chosen;
}
