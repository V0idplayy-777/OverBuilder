// Kernel layer: all heavy linear algebra goes through here.
// Preferred path is real WebAssembly (hand-written WAT in ./wat.ts, compiled
// in the browser at startup). If the WAT toolchain can't load (offline, CSP),
// we transparently fall back to equivalent typed-array JS kernels.

export type KernelMode = "boot" | "wasm" | "js";

let mode: KernelMode = "boot";
let wasm: {
  mem: WebAssembly.Memory;
  nt: (a: number, b: number, c: number, m: number, n: number, k: number, beta: number) => void;
  nn: (a: number, b: number, c: number, m: number, n: number, k: number, beta: number) => void;
} | null = null;

export function kernelMode(): KernelMode {
  return mode;
}

// ---------------------------------------------------------------- JS fallback
function jsNT(a: Float32Array, b: Float32Array, c: Float32Array, M: number, N: number, K: number, beta: number) {
  for (let i = 0; i < M; i++) {
    const ai = i * K;
    const ci = i * N;
    for (let j = 0; j < N; j += 4) {
      let s0 = beta * c[ci + j], s1 = beta * c[ci + j + 1], s2 = beta * c[ci + j + 2], s3 = beta * c[ci + j + 3];
      const bj = j * K;
      for (let k = 0; k < K; k++) {
        const av = a[ai + k];
        s0 += av * b[bj + k];
        s1 += av * b[bj + K + k];
        s2 += av * b[bj + 2 * K + k];
        s3 += av * b[bj + 3 * K + k];
      }
      c[ci + j] = s0; c[ci + j + 1] = s1; c[ci + j + 2] = s2; c[ci + j + 3] = s3;
    }
  }
}

function jsNN(a: Float32Array, b: Float32Array, c: Float32Array, M: number, N: number, K: number, beta: number) {
  for (let i = 0; i < M; i++) {
    const ai = i * K;
    const ci = i * N;
    for (let j = 0; j < N; j += 4) {
      let s0 = beta * c[ci + j], s1 = beta * c[ci + j + 1], s2 = beta * c[ci + j + 2], s3 = beta * c[ci + j + 3];
      for (let k = 0; k < K; k++) {
        const av = a[ai + k];
        const bk = k * N + j;
        s0 += av * b[bk];
        s1 += av * b[bk + 1];
        s2 += av * b[bk + 2];
        s3 += av * b[bk + 3];
      }
      c[ci + j] = s0; c[ci + j + 1] = s1; c[ci + j + 2] = s2; c[ci + j + 3] = s3;
    }
  }
}

// ---------------------------------------------------------------- wasm glue
let cursor = 0;
function wasmEnsure(bytes: number) {
  const mem = wasm!.mem;
  const need = cursor + bytes;
  if (need > mem.buffer.byteLength) {
    const pages = Math.ceil((need - mem.buffer.byteLength) / 65536);
    mem.grow(pages);
  }
}

function runWasm(fn: "nt" | "nn", a: Float32Array, b: Float32Array, c: Float32Array, M: number, N: number, K: number, beta: number) {
  const w = wasm!;
  const aB = a.length * 4, bB = b.length * 4, cB = c.length * 4;
  cursor = 16;
  const pa = cursor; cursor += aB;
  const pb = cursor; cursor += bB;
  const pc = cursor; cursor += cB;
  wasmEnsure(0);
  const buf = w.mem.buffer;
  new Float32Array(buf, pa, a.length).set(a);
  new Float32Array(buf, pb, b.length).set(b);
  if (beta !== 0) new Float32Array(buf, pc, c.length).set(c);
  (fn === "nt" ? w.nt : w.nn)(pa, pb, pc, M, N, K, beta);
  c.set(new Float32Array(buf, pc, c.length));
}

// ---------------------------------------------------------------- public API
type MatFn = (a: Float32Array, b: Float32Array, c: Float32Array, M: number, N: number, K: number, beta: number) => void;

function dispatch(nt: MatFn, nn: MatFn): { nt: MatFn; nn: MatFn } {
  if (wasm) {
    return {
      nt: (a, b, c, M, N, K, beta) => {
        if (M === 1 || M % 2 !== 0) { jsNT(a, b, c, M, N, K, beta); return; }
        runWasm("nt", a, b, c, M, N, K, beta);
      },
      nn: (a, b, c, M, N, K, beta) => {
        if (M === 1 || M % 2 !== 0) { jsNN(a, b, c, M, N, K, beta); return; }
        runWasm("nn", a, b, c, M, N, K, beta);
      },
    };
  }
  return { nt, nn };
}

export let matNT: MatFn = jsNT;
export let matNN: MatFn = jsNN;

const WASM_B64 = "AGFzbQEAAAABCwFgB39/f39/f38AAwMCAAAFBgEBAYCAAQcgAwZtZW1vcnkCAAhzZ2VtbV9udAAACHNnZW1tX25uAAEKzggC7AQCDH8PfSAGsiETIAVBAnQhESAEQQJ0IRJBACEHA0AgACAHIBFsaiEKIAogEWohC0EAIQgDQCACIAcgBGwgCGpBAnRqIRAgEyAQKgIAlCEaIBMgEEEEaioCAJQhGyATIBBBCGoqAgCUIRwgEyAQQQxqKgIAlCEdIBMgECASaioCAJQhHiATIBAgEkEEamoqAgCUIR8gEyAQIBJBCGpqKgIAlCEgIBMgECASQQxqaioCAJQhISABIAggEWxqIQwgDCARaiENIA0gEWohDiAOIBFqIQ9BACEJA0AgCioCACEUIApBBGoqAgAhFSALKgIAIRYgC0EEaioCACEXIAwqAgAhGCAMQQRqKgIAIRkgGiAUIBiUIBUgGZSSkiEaIB4gFiAYlCAXIBmUkpIhHiANKgIAIRggDUEEaioCACEZIBsgFCAYlCAVIBmUkpIhGyAfIBYgGJQgFyAZlJKSIR8gDioCACEYIA5BBGoqAgAhGSAcIBQgGJQgFSAZlJKSIRwgICAWIBiUIBcgGZSSkiEgIA8qAgAhGCAPQQRqKgIAIRkgHSAUIBiUIBUgGZSSkiEdICEgFiAYlCAXIBmUkpIhISAKQQhqIQogC0EIaiELIAxBCGohDCANQQhqIQ0gDkEIaiEOIA9BCGohDyAJQQJqIQkgCSAFSQ0ACyAQIBo4AgAgEEEEaiAbOAIAIBBBCGogHDgCACAQQQxqIB04AgAgECASaiAeOAIAIBAgEkEEamogHzgCACAQIBJBCGpqICA4AgAgECASQQxqaiAhOAIAIAhBBGohCCAIIARJDQALIAdBAmohByAHIANJDQALC90DAgl/D30gBrIhECAFQQJ0IQ4gBEECdCEPQQAhBwNAIAAgByAObGohCiAKIA5qIQtBACEIA0AgAiAHIARsIAhqQQJ0aiENIBAgDSoCAJQhFyAQIA1BBGoqAgCUIRggECANQQhqKgIAlCEZIBAgDUEMaioCAJQhGiAQIA0gD2oqAgCUIRsgECANIA9BBGpqKgIAlCEcIBAgDSAPQQhqaioCAJQhHSAQIA0gD0EMamoqAgCUIR4gASAIQQJ0aiEMQQAhCQNAIAogCUECdGoqAgAhESALIAlBAnRqKgIAIRIgDCoCACETIAxBBGoqAgAhFCAMQQhqKgIAIRUgDEEMaioCACEWIBcgESATlJIhFyAYIBEgFJSSIRggGSARIBWUkiEZIBogESAWlJIhGiAbIBIgE5SSIRsgHCASIBSUkiEcIB0gEiAVlJIhHSAeIBIgFpSSIR4gDCAPaiEMIAlBAWohCSAJIAVJDQALIA0gFzgCACANQQRqIBg4AgAgDUEIaiAZOAIAIA1BDGogGjgCACANIA9qIBs4AgAgDSAPQQRqaiAcOAIAIA0gD0EIamogHTgCACANIA9BDGpqIB44AgAgCEEEaiEIIAggBEkNAAsgB0ECaiEHIAcgA0kNAAsL";

function b64ToBuf(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

export let kernelError = "";

export async function initKernels(): Promise<KernelMode> {
  try {
    const bin = b64ToBuf(WASM_B64);
    const { instance } = await WebAssembly.instantiate(bin, {});
    const ex: any = instance.exports;
    wasm = { mem: ex.memory, nt: ex.sgemm_nt, nn: ex.sgemm_nn };
    const M = 8, N = 12, K = 16;
    const rnd = (n: number) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
    const a = rnd(M * K), b = rnd(N * K), bT = rnd(K * N);
    const c1 = new Float32Array(M * N), c2 = new Float32Array(M * N);
    jsNT(a, b, c1, M, N, K, 0);
    runWasm("nt", a, b, c2, M, N, K, 0);
    let d = 0; for (let i = 0; i < c1.length; i++) d = Math.max(d, Math.abs(c1[i] - c2[i]));
    const d1 = new Float32Array(M * N), d2 = new Float32Array(M * N);
    jsNN(a, bT as Float32Array, d1, M, N, K, 0);
    runWasm("nn", a, bT as Float32Array, d2, M, N, K, 0);
    let dd = 0; for (let i = 0; i < d1.length; i++) dd = Math.max(dd, Math.abs(d1[i] - d2[i]));
    if (d > 1e-4 || dd > 1e-4) throw new Error("wasm self-test failed");
    const fns = dispatch(jsNT, jsNN);
    matNT = fns.nt; matNN = fns.nn;
    mode = "wasm";
  } catch (e) {
    console.warn("[kernels] WASM unavailable, using JS kernels:", e);
    wasm = null;
    matNT = jsNT; matNN = jsNN;
    mode = "js";
  }
  return mode;
}

// matrix-vector product for single-token decode (memory-bound; JS is ideal)
export function matVec(x: Float32Array, w: Float32Array, out: Float32Array, outDim: number, inDim: number) {
  for (let o = 0; o < outDim; o++) {
    const wr = o * inDim;
    let s = 0;
    for (let i = 0; i < inDim; i += 4) {
      s += x[i] * w[wr + i] + x[i + 1] * w[wr + i + 1] + x[i + 2] * w[wr + i + 2] + x[i + 3] * w[wr + i + 3];
    }
    out[o] = s;
  }
}

export function transpose(src: Float32Array, dst: Float32Array, rows: number, cols: number) {
  for (let r = 0; r < rows; r++) {
    const ro = r * cols;
    for (let c = 0; c < cols; c++) dst[c * rows + r] = src[ro + c];
  }
}
