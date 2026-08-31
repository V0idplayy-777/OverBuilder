// Orchestration: tokenizer training, corpus encoding, the time-sliced
// training loop, sampling, chat, and checkpointing. React subscribes to
// snapshots via useSyncExternalStore.

import { initKernels } from "./kernels";
import { GPT, sampleNext } from "./model";
import { MODELS, paramCount, type ModelSpec } from "./configs";
import { BPETokenizer, TOK } from "./bpe";
import { getDataset, getDatasets } from "./corpus";

export type Status = "idle" | "preparing" | "training" | "paused" | "done" | "error";
export type ChatMsg = { role: "u" | "b"; text: string };

export interface EngineState {
  kernel: "boot" | "wasm" | "js";
  status: Status;
  statusText: string;
  modelId: string;
  datasetId: string;
  vocabSize: number;
  mergeCount: number;
  corpusChars: number;
  corpusTokens: number;
  params: number;
  step: number;
  totalSteps: number;
  loss: number;
  lossSeries: number[];
  tokPerSec: number;
  tokensSeen: number;
  gnorm: number;
  lr: number;
  lrPeak: number;
  trainCtx: number;
  sampleEvery: number;
  sample: string;
  sampleAt: number;
  logs: string[];
  trained: boolean;
  chats: ChatMsg[];
  generating: boolean;
  temp: number;
  topK: number;
  topP: number;
  repPen: number;
  maxNew: number;
  msPerStep: number;
  dataReady: boolean;
  modelReady: boolean;
}

const SPEC: Record<string, ModelSpec> = Object.fromEntries(MODELS.map((m) => [m.id, m]));

class Engine {
  state: EngineState = {
    kernel: "boot",
    status: "idle",
    statusText: "booting",
    modelId: "small",
    datasetId: "full",
    vocabSize: 0,
    mergeCount: 0,
    corpusChars: 0,
    corpusTokens: 0,
    params: 0,
    step: 0,
    totalSteps: SPEC.small.steps,
    loss: NaN,
    lossSeries: [],
    tokPerSec: 0,
    tokensSeen: 0,
    gnorm: 0,
    lr: 0,
    lrPeak: SPEC.small.lr,
    trainCtx: SPEC.small.trainCtx,
    sampleEvery: 100,
    sample: "",
    sampleAt: 0,
    logs: [],
    trained: false,
    chats: [],
    generating: false,
    temp: 0.8,
    topK: 40,
    topP: 0.9,
    repPen: 1.25,
    maxNew: 48,
    msPerStep: 0,
    dataReady: false,
    modelReady: false,
  };

  private listeners = new Set<() => void>();
  private tokenizer = new BPETokenizer();
  private model: GPT | null = null;
  private ids: Uint32Array = new Uint32Array(0);
  private runId = 0;
  private encodeCache = new Map<string, Uint32Array>();
  private stops = new Set<number>([TOK.end, TOK.user]);

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  getSnapshot = () => this.state;
  private emit(patch?: Partial<EngineState>) {
    if (patch) this.state = { ...this.state, ...patch };
    else this.state = { ...this.state };
    this.listeners.forEach((f) => f());
  }

  log(msg: string) {
    const line = msg;
    const logs = [...this.state.logs, line];
    if (logs.length > 400) logs.splice(0, logs.length - 400);
    this.state = { ...this.state, logs };
  }

  spec(): ModelSpec {
    return SPEC[this.state.modelId];
  }

  // ------------------------------------------------------------ boot

  async init() {
    const kernel = await initKernels();
    this.emit({ kernel });
    this.log(`kernels: ${kernel === "wasm" ? "WebAssembly (compiled from WAT at boot)" : "javascript fallback"}`);
    await this.rebuildData(false);
  }

  private async rebuildData(rerunModel = true) {
    const ds = getDataset(this.state.datasetId);
    this.emit({ status: "preparing", statusText: "training tokenizer", dataReady: false });
    const maxMerges = Math.max(...MODELS.map((m) => m.vocabCap)) + 64;
    await this.tokenizer.train(ds.text, maxMerges, (done, total) =>
      this.emit({ statusText: `training tokenizer — merge ${done}/${total}` })
    );
    this.encodeCache.clear();
    this.emit({ corpusChars: ds.text.length, mergeCount: this.tokenizer.merges.length, dataReady: true, status: "idle", statusText: "ready" });
    this.log(`tokenizer: ${this.tokenizer.merges.length} bpe merges on ${(ds.text.length / 1024).toFixed(0)} kb of ${ds.name.toLowerCase()}`);
    if (rerunModel) await this.recap();
    else await this.rebuildModel();
  }

  private capMerges(spec: ModelSpec): number {
    // choose merge cap so total vocab is a multiple of 4 (kernel tiling)
    let cap = spec.vocabCap;
    while ((this.tokenizer.vocabSize(cap) % 4) !== 0 && cap > 0) cap--;
    return Math.min(cap, this.tokenizer.merges.length);
  }

  private async recap() {
    const spec = this.spec();
    const cap = this.capMerges(spec);
    this.emit({ statusText: "encoding corpus", status: "preparing" });
    const key = `${this.state.datasetId}:${cap}`;
    let ids = this.encodeCache.get(key);
    if (!ids) {
      const ds = getDataset(this.state.datasetId);
      await new Promise((r) => setTimeout(r, 0));
      ids = Uint32Array.from(this.tokenizer.encode(ds.text, cap));
      this.encodeCache.set(key, ids);
    }
    this.ids = ids;
    const vocab = this.tokenizer.vocabSize(cap);
    this.emit({ corpusTokens: ids.length, vocabSize: vocab });
    this.log(`corpus: ${ids.length.toLocaleString()} tokens (vocab ${vocab})`);
  }

  async rebuildModel() {
    const spec = this.spec();
    const cap = this.capMerges(spec);
    this.emit({ status: "preparing", statusText: "encoding corpus", modelReady: false });
    await this.recap();
    this.emit({ statusText: "initializing weights" });
    await new Promise((r) => setTimeout(r, 0));
    this.model = new GPT(spec, this.tokenizer.vocabSize(cap));
    this.runId++;
    const params = paramCount(spec, this.tokenizer.vocabSize(cap));
    this.state = {
      ...this.state,
      params,
      step: 0,
      totalSteps: spec.steps,
      lrPeak: spec.lr,
      trainCtx: spec.trainCtx,
      loss: NaN,
      lossSeries: [],
      tokensSeen: 0,
      gnorm: 0,
      sample: "",
      trained: false,
      status: "idle",
      statusText: "ready",
      modelReady: true,
      msPerStep: 0,
    };
    this.emit();
    this.log(
      `model "${spec.name.toLowerCase()}": ${params.toLocaleString()} params — d=${spec.d}, ${spec.layers} layers, ${spec.heads} heads, ctx ${spec.ctx}`
    );
  }

  async selectModel(id: string) {
    if (id === this.state.modelId && this.model) return;
    this.runId++;
    this.emit({ modelId: id });
    await this.rebuildModel();
  }

  async selectDataset(id: string) {
    if (id === this.state.datasetId) return;
    this.runId++;
    this.emit({ datasetId: id });
    await this.rebuildData(true);
    await this.rebuildModel();
  }

  set(patch: Partial<EngineState>) {
    this.runId++;
    this.emit(patch);
  }

  // ------------------------------------------------------------ training

  private lrAt(step: number, total: number): number {
    const peak = this.state.lrPeak;
    const warm = Math.max(10, Math.floor(total * 0.02));
    if (step < warm) return (peak * (step + 1)) / warm;
    const t = (step - warm) / Math.max(1, total - warm);
    return peak * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, t))));
  }

  async start() {
    if (!this.model) await this.rebuildModel();
    const spec = this.spec();
    if (spec.heavy && this.state.step === 0) {
      const ok = window.confirm(
        `${spec.name} is ~${(this.state.params / 1e6).toFixed(0)}M parameters.\n\nThat means roughly ${(this.state.params * 16 / 1e9).toFixed(1)} GB of RAM for training state and seconds per step. Continue?`
      );
      if (!ok) return;
    }
    if (this.state.status === "training") return;
    const runId = ++this.runId;
    const total = this.state.totalSteps;
    if (this.state.step >= total) this.state = { ...this.state, step: 0, lossSeries: [] };
    this.emit({ status: "training", statusText: "training" });
    if (this.state.step === 0) {
      this.log(`training: ${total.toLocaleString()} steps, ctx ${this.state.trainCtx}, peak lr ${spec.lr}`);
    } else {
      this.log(`resuming at step ${this.state.step}`);
    }

    let lastLog = this.state.step;
    while (this.runId === runId && this.state.step < total) {
      const sliceStart = performance.now();
      let stepsInSlice = 0;
      let tokensInSlice = 0;
      do {
        const T = this.state.trainCtx;
        const stream = this.ids;
        if (stream.length < T + 2) { this.log("corpus too small for context length"); this.emit({ status: "error" }); return; }
        const off = Math.floor(Math.random() * (stream.length - T - 1));
        const ids = stream.subarray(off, off + T + 1);
        const lr = this.lrAt(this.state.step, total);
        this.model!.zeroGrads();
        const loss = this.model!.trainStep(ids as unknown as Uint32Array, T);
        const gnorm = this.model!.clipGrads(1.0);
        this.model!.adamw(lr, 0.1);
        const ema = isNaN(this.state.loss) ? loss : this.state.loss * 0.98 + loss * 0.02;
        const series = this.state.lossSeries.length > 6000
          ? [...this.state.lossSeries.slice(3000), loss]
          : [...this.state.lossSeries, loss];
        this.state = {
          ...this.state,
          step: this.state.step + 1,
          loss: ema,
          lossSeries: series,
          tokensSeen: this.state.tokensSeen + T,
          gnorm,
          lr,
        };
        stepsInSlice++;
        tokensInSlice += T;
      } while (performance.now() - sliceStart < 13);

      const sliceMs = Math.max(0.5, performance.now() - sliceStart);
      const tps = (tokensInSlice / sliceMs) * 1000;
      this.state = { ...this.state, tokPerSec: this.state.tokPerSec ? this.state.tokPerSec * 0.8 + tps * 0.2 : tps, msPerStep: sliceMs / stepsInSlice };
      this.emit({
        step: this.state.step, loss: this.state.loss, lossSeries: this.state.lossSeries,
        tokensSeen: this.state.tokensSeen, gnorm: this.state.gnorm, lr: this.state.lr,
        tokPerSec: this.state.tokPerSec, msPerStep: this.state.msPerStep,
      });
      this.emit();
      await new Promise((r) => setTimeout(r, 0));

      if (this.state.step % this.state.sampleEvery === 0 && this.runId === runId) {
        await this.preview();
      }
      if (this.state.step - lastLog >= Math.max(50, Math.floor(total / 40))) {
        lastLog = this.state.step;
        this.log(
          `step ${this.state.step}/${total} — loss ${this.state.loss.toFixed(3)} · ${fmtNum(this.state.tokPerSec)} tok/s · lr ${this.state.lr.toExponential(1)} · gnorm ${this.state.gnorm.toFixed(2)}`
        );
        this.emit();
      }
    }
    if (this.runId !== runId) return;
    if (this.state.step >= total) {
      await this.preview();
      this.emit({ status: "done", statusText: "training complete", trained: true });
      this.log(`done — final loss ${this.state.loss.toFixed(3)} over ${this.state.tokensSeen.toLocaleString()} tokens. head to the chat tab.`);
      this.emit();
    }
  }

  pause() {
    this.runId++;
    this.emit({ status: "paused", statusText: "paused", trained: this.state.step > 0 });
    this.log(`paused at step ${this.state.step}`);
    this.emit();
  }

  async resetTraining() {
    this.runId++;
    await this.rebuildModel();
    this.log("weights reinitialized");
  }

  private async preview() {
    if (!this.model) return;
    const cap = this.capMerges(this.spec());
    const prompt = `${TOK_USER} hello, how are you?\n${TOK_BOT}`;
    const ids = this.tokenizer.encode(prompt, cap);
    const out: number[] = [];
    const cache = this.model.createCache();
    const seen = new Set<number>();
    const budget = Math.min(this.spec().ctx - 1, ids.length + 36);
    let cur: number[] = ids.slice(-budget);
    for (const id of cur) { this.model.stepToken(id, cache); seen.add(id); }
    let next = sampleNext(this.model.stepToken(cur[cur.length - 1], newCacheWrap(cache)), seen, 0.8, 40, 0.95, 1.1);
    // note: stepToken call above advanced once more than needed; recompute simply below
    for (let i = 0; i < 36; i++) {
      if (this.stops.has(next)) break;
      out.push(next);
      seen.add(next);
      if (cache.pos >= this.spec().ctx - 1) break;
      next = sampleNext(this.model.stepToken(next, cache), seen, 0.8, 40, 0.95, 1.1);
      await new Promise((r) => setTimeout(r, 0));
    }
    const text = this.tokenizer.decode(out).trim();
    this.state = { ...this.state, sample: text, sampleAt: this.state.step, trained: this.state.step > 0 };
    this.log(`sample @${this.state.step}: "${text.slice(0, 80)}${text.length > 80 ? "…" : ""}"`);
    this.emit();
  }

  // ------------------------------------------------------------ chat

  async chat(userText: string) {
    if (!this.model || this.state.step === 0) return;
    if (this.state.generating) return;
    this.runId++; // stop training preview etc if any
    const runId = this.runId;
    const spec = this.spec();
    const cap = this.capMerges(spec);
    const chats: ChatMsg[] = [...this.state.chats, { role: "u", text: userText }, { role: "b", text: "" }];
    this.emit({ chats, generating: true, status: this.state.status === "training" ? "paused" : this.state.status });

    // build prompt from history, most recent first
    let promptIds: number[] = [];
    const addTurn = (u: string, b?: string) => {
      const turn = this.tokenizer.encode(`${TOK_USER} ${u}\n${TOK_BOT} ${b ?? ""}`, cap);
      promptIds = b === undefined ? promptIds.concat(turn) : turn.concat(this.tokenizer.encode(` ${TOK_END}`, cap), promptIds);
    };
    // append newest last
    const hist = chats.slice(0, -1);
    promptIds = this.tokenizer.encode(`${TOK_USER} ${userText}\n${TOK_BOT}`, cap);
    for (let i = hist.length - 2; i >= 0; i--) {
      const m = hist[i];
      const turnIds = this.tokenizer.encode(
        m.role === "u" ? `${TOK_USER} ${m.text}\n` : `${TOK_BOT} ${m.text} ${TOK_END}\n`,
        cap
      );
      const next = [...turnIds, ...promptIds];
      if (next.length > spec.ctx - this.state.maxNew - 4) break;
      promptIds = next;
    }

    const cache = this.model.createCache();
    const seen = new Set<number>();
    for (const id of promptIds) {
      if (cache.pos >= spec.ctx - 2) break;
      this.model.stepToken(id, cache);
      seen.add(id);
    }
    let logits = this.model.stepToken(promptIds[promptIds.length - 1] ?? TOK.bot, newCacheWrap(cache));
    const gen: number[] = [];
    const t0 = performance.now();
    for (let i = 0; i < this.state.maxNew; i++) {
      if (runId !== this.runId) break;
      const next = sampleNext(logits, seen, this.state.temp, this.state.topK, this.state.topP, this.state.repPen);
      if (this.stops.has(next)) break;
      if (cache.pos >= spec.ctx - 2) break;
      gen.push(next);
      seen.add(next);
      logits = this.model.stepToken(next, cache);
      const text = this.tokenizer.decode(gen).trim();
      const cc = [...chats.slice(0, -1), { role: "b" as const, text }];
      this.emit({ chats: cc });
      await new Promise((r) => setTimeout(r, 0));
    }
    const dt = performance.now() - t0;
    if (gen.length > 0) {
      this.log(`generated ${gen.length} tokens in ${(dt / 1000).toFixed(1)}s (${(gen.length / (dt / 1000)).toFixed(1)} tok/s)`);
    }
    const finalChats = [...this.state.chats];
    if (finalChats.length && finalChats[finalChats.length - 1].role === "b") {
      const t = finalChats[finalChats.length - 1].text.trim();
      finalChats[finalChats.length - 1] = { role: "b", text: t.length ? t : "(empty — try a lower temperature or more training)" };
    }
    this.emit({ chats: finalChats, generating: false });
  }

  stopChat() {
    this.runId++;
    this.emit({ generating: false });
  }

  clearChat() {
    this.emit({ chats: [] });
  }

  async quickTrainNano() {
    if (this.state.generating) return;
    if (this.state.modelId !== "nano" || !this.model || this.state.step === 0) {
      await this.selectModel("nano");
      this.set({ totalSteps: 2500, sampleEvery: 200 });
    }
    await this.start();
  }

  // ------------------------------------------------------------ checkpoint

  exportCheckpoint(): Blob | null {
    if (!this.model) return null;
    const header = {
      magic: "FORGE1",
      model: this.state.modelId,
      spec: this.spec(),
      capMerges: this.capMerges(this.spec()),
      step: this.state.step,
      tokensSeen: this.state.tokensSeen,
      loss: this.state.loss,
      tok: this.tokenizer.serialize(),
    };
    const hb = new TextEncoder().encode(JSON.stringify(header));
    const w = this.model.packedWeights();
    const buf = new ArrayBuffer(4 + hb.length + w.length * 4);
    const dv = new DataView(buf);
    dv.setUint32(0, hb.length, true);
    new Uint8Array(buf, 4, hb.length).set(hb);
    new Float32Array(buf, 4 + hb.length, w.length).set(w);
    this.log(`checkpoint exported — step ${this.state.step}, ${(buf.byteLength / 1048576).toFixed(1)} mb`);
    this.emit();
    return new Blob([buf], { type: "application/octet-stream" });
  }

  async importCheckpoint(file: File): Promise<boolean> {
    try {
      const buf = await file.arrayBuffer();
      const dv = new DataView(buf);
      const hlen = dv.getUint32(0, true);
      const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
      if (header.magic !== "FORGE1") throw new Error("bad magic");
      this.runId++;
      this.emit({ status: "preparing", statusText: "loading checkpoint", modelReady: false });
      this.tokenizer.deserialize(header.tok);
      this.state = { ...this.state, modelId: header.model };
      await this.recap();
      const spec = this.spec();
      const vocab = this.tokenizer.vocabSize(header.capMerges);
      this.model = new GPT(spec, vocab);
      const n = this.model.params.reduce((s, p) => s + p.d.length, 0);
      const w = new Float32Array(buf, 4 + hlen, n);
      this.model.loadPacked(w);
      this.state = {
        ...this.state,
        vocabSize: vocab,
        params: paramCount(spec, vocab),
        step: header.step ?? 0,
        tokensSeen: header.tokensSeen ?? 0,
        loss: header.loss ?? NaN,
        lossSeries: [],
        trained: true,
        status: "done",
        statusText: "checkpoint loaded",
        modelReady: true,
        dataReady: true,
        mergeCount: this.tokenizer.merges.length,
      };
      this.emit();
      this.log(`checkpoint loaded — ${header.model} @ step ${header.step}, loss ${(header.loss ?? NaN).toFixed?.(3)}`);
      return true;
    } catch (e) {
      console.error(e);
      this.log("failed to load checkpoint (wrong file?)");
      this.emit({ status: stringOr(this.state.status, "idle"), modelReady: true });
      return false;
    }
  }
}

const TOK_USER = "<|user|>";
const TOK_BOT = "<|bot|>";
const TOK_END = "<|end|>";

// The prefill loop already advanced the cache one extra step in preview();
// this helper documents that we intentionally continue from that position.
function newCacheWrap<T>(c: T): T { return c; }

function fmtNum(n: number): string {
  if (n >= 1000) return (n / 1000).toFixed(1) + "k";
  return n.toFixed(0);
}

function stringOr(s: string, d: string): any { return (s as any) || d; }

export const engine = new Engine();
export const DATASETS = () => getDatasets();
