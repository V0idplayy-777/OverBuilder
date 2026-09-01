// Orchestration: tokenizer training, corpus encoding, the time-sliced
// training loop, sampling, chat, and checkpointing. React subscribes to
// snapshots via useSyncExternalStore.

import { initKernels, kernelError } from "./nn/kernels";
import { GPT, sampleNext } from "./nn/model";
import { MODELS, paramCount, type ModelSpec } from "./nn/configs";
import { BPETokenizer, TOK } from "./tok/bpe";
import { getDataset, getDatasets } from "./data/corpus";

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
const SPECIAL_RE = /(<\|user\|>|<\|bot\|>|<\|end\|>)/;
const SPECIAL_ID: Record<string, number> = { "<|user|>": TOK.user, "<|bot|>": TOK.bot, "<|end|>": TOK.end };

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
  private emit() {
    this.state = { ...this.state };
    this.listeners.forEach((f) => f());
  }
  private set(patch: Partial<EngineState>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((f) => f());
  }

  log(msg: string) {
    const logs = [...this.state.logs, msg];
    if (logs.length > 400) logs.splice(0, logs.length - 400);
    this.state = { ...this.state, logs };
    this.listeners.forEach((f) => f());
  }

  spec(): ModelSpec {
    return SPEC[this.state.modelId];
  }

  // encode text, splicing real special-token ids in place of their markers
  private encodeSmart(text: string, cap: number): number[] {
    const out: number[] = [];
    for (const part of text.split(SPECIAL_RE)) {
      if (!part) continue;
      const sid = SPECIAL_ID[part];
      if (sid !== undefined) out.push(sid);
      else out.push(...this.tokenizer.encode(part, cap));
    }
    return out;
  }

  // ------------------------------------------------------------ boot

  async init() {
    const kernel = await initKernels();
    this.set({ kernel });
    this.log(kernel === "wasm" ? "kernels: wasm" : "kernels: javascript fallback");
    if (kernelError) this.log("wasm error: " + kernelError);
    await this.rebuildData();
    await this.rebuildModel();
  }

  private async rebuildData() {
    const ds = getDataset(this.state.datasetId);
    this.set({ status: "preparing", statusText: "training tokenizer", dataReady: false });
    const maxMerges = Math.max(...MODELS.map((m) => m.vocabCap)) + 64;
    await this.tokenizer.train(ds.text, maxMerges, (done, total) =>
      this.set({ statusText: `training tokenizer — merge ${done}/${total}` })
    );
    this.encodeCache.clear();
    this.set({ corpusChars: ds.text.length, mergeCount: this.tokenizer.merges.length, dataReady: true });
    this.log(`tokenizer: ${this.tokenizer.merges.length} bpe merges over ${(ds.text.length / 1024).toFixed(0)} kb — ${ds.name.toLowerCase()}`);
  }

  private capMerges(spec: ModelSpec): number {
    let cap = spec.vocabCap;
    while (this.tokenizer.vocabSize(cap) % 4 !== 0 && cap > 0) cap--;
    return Math.min(cap, this.tokenizer.merges.length);
  }

  private async recap(): Promise<number> {
    const spec = this.spec();
    const cap = this.capMerges(spec);
    this.set({ statusText: "encoding corpus", status: "preparing" });
    await new Promise((r) => setTimeout(r, 0));
    const key = `${this.state.datasetId}:${cap}`;
    let ids = this.encodeCache.get(key);
    if (!ids) {
      const ds = getDataset(this.state.datasetId);
      ids = Uint32Array.from(this.encodeSmart(ds.text, cap));
      this.encodeCache.set(key, ids);
    }
    this.ids = ids;
    const vocab = this.tokenizer.vocabSize(cap);
    this.set({ corpusTokens: ids.length, vocabSize: vocab });
    this.log(`corpus: ${ids.length.toLocaleString()} tokens, vocab ${vocab.toLocaleString()}`);
    return vocab;
  }

  async rebuildModel() {
    const spec = this.spec();
    this.runId++;
    this.set({ status: "preparing", modelReady: false });
    const vocab = await this.recap();
    this.set({ statusText: "initializing weights" });
    await new Promise((r) => setTimeout(r, 0));
    this.model = new GPT(spec, vocab);
    const params = paramCount(spec, vocab);
    this.set({
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
      tokPerSec: 0,
    });
    this.log(
      `model "${spec.name.toLowerCase()}": ${params.toLocaleString()} params — d=${spec.d}, ${spec.layers} layers × ${spec.heads} heads, ctx ${spec.ctx}, vocab ${vocab}`
    );
  }

  async selectModel(id: string) {
    if (id === this.state.modelId && this.model) return;
    this.runId++;
    this.set({ modelId: id });
    await this.rebuildModel();
  }

  async selectDataset(id: string) {
    if (id === this.state.datasetId) return;
    this.runId++;
    this.set({ datasetId: id });
    await this.rebuildData();
    await this.rebuildModel();
  }

  // live-applies hyperparameters / sampling settings without interrupting a run
  update(patch: Partial<EngineState>) {
    this.set(patch);
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
        `${spec.name} is ~${(this.state.params / 1e6).toFixed(0)}M parameters.\n\nThat's roughly ${((this.state.params * 16) / 1e9).toFixed(1)} GB of RAM for optimizer state, and seconds (not milliseconds) per step. Continue?`
      );
      if (!ok) return;
    }
    if (this.state.status === "training") return;
    const runId = ++this.runId;
    const total = this.state.totalSteps;
    if (this.state.step >= total) this.set({ step: 0, lossSeries: [] });
    const fresh = this.state.step === 0;
    this.set({ status: "training", statusText: "training" });
    this.log(
      fresh
        ? `training: ${total.toLocaleString()} steps · ctx ${this.state.trainCtx} · peak lr ${spec.lr} · adamw + cosine`
        : `resuming at step ${this.state.step}`
    );

    let lastLog = this.state.step;
    while (this.runId === runId && this.state.step < total) {
      const sliceStart = performance.now();
      let stepsInSlice = 0;
      let tokensInSlice = 0;
      do {
        const T = this.state.trainCtx;
        const stream = this.ids;
        if (stream.length < T + 2) {
          this.log("corpus too small for this context length");
          this.set({ status: "error" });
          return;
        }
        const off = Math.floor(Math.random() * (stream.length - T - 1));
        const window = stream.subarray(off, off + T + 1);
        const lr = this.lrAt(this.state.step, total);
        this.model!.zeroGrads();
        const loss = this.model!.trainStep(window as Uint32Array, T);
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
      } while (performance.now() - sliceStart < 28);

      const sliceMs = Math.max(0.5, performance.now() - sliceStart);
      const tps = (tokensInSlice / sliceMs) * 1000;
      this.state = {
        ...this.state,
        tokPerSec: this.state.tokPerSec ? this.state.tokPerSec * 0.8 + tps * 0.2 : tps,
        msPerStep: sliceMs / stepsInSlice,
      };
      this.emit();
      await new Promise((r) => setTimeout(r, 0));
      if (this.runId !== runId) return;

      if (this.state.step % this.state.sampleEvery === 0) {
        await this.preview();
        if (this.runId !== runId) return;
      }
      if (this.state.step - lastLog >= Math.max(50, Math.floor(total / 40))) {
        lastLog = this.state.step;
        this.log(
          `step ${this.state.step}/${total} — loss ${this.state.loss.toFixed(3)} · ${fmtNum(this.state.tokPerSec)} tok/s · lr ${this.state.lr.toExponential(1)} · gnorm ${this.state.gnorm.toFixed(2)}`
        );
      }
    }
    if (this.state.step >= total) {
      await this.preview();
      this.set({ status: "done", statusText: "training complete", trained: true });
      this.log(`done — final loss ${this.state.loss.toFixed(3)} over ${this.state.tokensSeen.toLocaleString()} tokens. head to the chat tab.`);
    }
  }

  pause() {
    this.runId++;
    this.set({ status: "paused", statusText: "paused", trained: this.state.step > 0 });
    this.log(`paused at step ${this.state.step}`);
  }

  async resetTraining() {
    this.runId++;
    await this.rebuildModel();
    this.log("weights reinitialized");
  }

  private async preview() {
    if (!this.model) return;
    const cap = this.capMerges(this.spec());
    const ids = this.encodeSmart("<|user|> hello, how are you?\n<|bot|>", cap);
    const out: number[] = [];
    const cache = this.model.createCache();
    const seen = new Set<number>();
    let logits: Float32Array | null = null;
    for (const id of ids) {
      if (cache.pos >= this.spec().ctx - 2) break;
      logits = this.model.stepToken(id, cache);
      seen.add(id);
    }
    for (let i = 0; i < 36 && logits; i++) {
      const next = sampleNext(logits, seen, 0.8, 40, 0.95, 1.1);
      if (this.stops.has(next)) break;
      out.push(next);
      seen.add(next);
      if (cache.pos >= this.spec().ctx - 2) break;
      logits = this.model.stepToken(next, cache);
    }
    const text = this.tokenizer.decode(out).trim();
    this.state = { ...this.state, sample: text, sampleAt: this.state.step, trained: this.state.step > 0 };
    if (this.state.step % (this.state.sampleEvery * 5) === 0 && text) {
      this.log(`sample @${this.state.step}: "${text.slice(0, 90)}${text.length > 90 ? "…" : ""}"`);
    }
    this.emit();
  }

  // ------------------------------------------------------------ chat

  async chat(userText: string) {
    if (!this.model || this.state.step === 0 || this.state.generating) return;
    const wasTraining = this.state.status === "training";
    this.runId++;
    const runId = this.runId;
    const spec = this.spec();
    const cap = this.capMerges(spec);
    const base: ChatMsg[] = [...this.state.chats, { role: "u", text: userText }];
    this.set({ chats: [...base, { role: "b", text: "" }], generating: true, status: wasTraining ? "paused" : this.state.status });

    // prompt: newest message last, older turns prepended while they fit
    let promptIds = this.encodeSmart(`<|user|> ${userText}\n<|bot|>`, cap);
    const hist = base.slice(0, -1);
    for (let i = hist.length - 1; i >= 0; i--) {
      const m = hist[i];
      const turn = this.encodeSmart(
        m.role === "u" ? `<|user|> ${m.text}\n` : `<|bot|> ${m.text} <|end|>\n`,
        cap
      );
      const joined = [...turn, ...promptIds];
      if (joined.length > spec.ctx - this.state.maxNew - 4) break;
      promptIds = joined;
    }

    const cache = this.model.createCache();
    const seen = new Set<number>();
    let logits: Float32Array | null = null;
    for (const id of promptIds) {
      if (cache.pos >= spec.ctx - 2) break;
      logits = this.model.stepToken(id, cache);
      seen.add(id);
    }
    const gen: number[] = [];
    const t0 = performance.now();
    for (let i = 0; i < this.state.maxNew && logits; i++) {
      if (runId !== this.runId) break;
      const next = sampleNext(logits, seen, this.state.temp, this.state.topK, this.state.topP, this.state.repPen);
      if (this.stops.has(next)) break;
      if (cache.pos >= spec.ctx - 2) break;
      gen.push(next);
      seen.add(next);
      logits = this.model.stepToken(next, cache);
      const text = this.tokenizer.decode(gen).trim();
      this.set({ chats: [...base, { role: "b", text }] });
      await new Promise((r) => setTimeout(r, 0));
    }
    const dt = (performance.now() - t0) / 1000;
    if (gen.length > 1) this.log(`generated ${gen.length} tokens in ${dt.toFixed(1)}s (${(gen.length / dt).toFixed(1)} tok/s)`);
    const finalChats = [...this.state.chats];
    const last = finalChats[finalChats.length - 1];
    if (last?.role === "b" && !last.text.trim()) {
      finalChats[finalChats.length - 1] = { role: "b", text: "(nothing useful yet — try more steps or lower temperature)" };
    }
    this.set({ chats: finalChats, generating: false });
  }

  stopChat() {
    this.runId++;
    this.set({ generating: false });
  }

  clearChat() {
    this.set({ chats: [] });
  }

  async quickTrain() {
    if (this.state.generating) return;
    if (!this.model || this.state.modelId !== "nano" || this.state.step === 0) {
      await this.selectModel("nano");
      this.update({ totalSteps: 2500, sampleEvery: 500 });
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
      dataset: this.state.datasetId,
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
    return new Blob([buf], { type: "application/octet-stream" });
  }

  async importCheckpoint(file: File): Promise<boolean> {
    try {
      const buf = await file.arrayBuffer();
      const dv = new DataView(buf);
      const hlen = dv.getUint32(0, true);
      const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
      if (header.magic !== "FORGE1") throw new Error("not a forge checkpoint");
      this.runId++;
      this.set({ status: "preparing", statusText: "loading checkpoint", modelReady: false });
      this.tokenizer.deserialize(header.tok);
      this.state = { ...this.state, modelId: header.model, datasetId: header.dataset ?? this.state.datasetId };
      const cap = header.capMerges;
      const vocab = this.tokenizer.vocabSize(cap);
      this.ids = this.encodeSmart ? Uint32Array.from(this.encodeSmart(getDataset(this.state.datasetId).text, cap)) : new Uint32Array(0);
      this.model = new GPT(this.spec(), vocab);
      const n = this.model.params.reduce((s, p) => s + p.d.length, 0);
      this.model.loadPacked(new Float32Array(buf, 4 + hlen, n));
      this.state = {
        ...this.state,
        vocabSize: vocab,
        corpusTokens: this.ids.length,
        params: paramCount(this.spec(), vocab),
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
      this.log(`checkpoint loaded — ${header.model} @ step ${header.step}, loss ${(header.loss ?? NaN).toFixed(3)}`);
      return true;
    } catch (e) {
      console.error(e);
      this.log("couldn't load that file (not a forge checkpoint?)");
      this.set({ status: "idle", statusText: "ready", modelReady: true });
      return false;
    }
  }
}

function fmtNum(n: number): string {
  if (n >= 1000) return (n / 1000).toFixed(1) + "k";
  return n.toFixed(0);
}

export const engine = new Engine();
export const DATASETS = () => getDatasets();
