// Byte-pair encoding (sub-word) tokenizer, trained in the browser on the
// active dataset. Special tokens first, then every seen character, then
// merge-derived sub-words in merge (frequency) order — so truncating the
// merge list gives a principled smaller vocabulary for the tiny models.

export const SPECIALS = ["<|pad|>", "<|end|>", "<|user|>", "<|bot|>"] as const;
export const TOK = { pad: 0, end: 1, user: 2, bot: 3 };

const PRETOK = /\s+|[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z\d]/g;

type WordId = number;

export class BPETokenizer {
  chars: string[] = [];
  merges: [string, string][] = [];
  private mergeRank = new Map<string, number>();
  private tokenToId = new Map<string, number>();
  private idToToken: string[] = [];
  baseVocab = 0;
  trained = false;

  async train(corpus: string, mergeCount: number, onProgress?: (done: number, total: number) => void) {
    const t0 = performance.now();

    // --- pretokenize + count
    const freq = new Map<string, number>();
    const seenChars = new Set<string>();
    corpus = corpus + "\n " + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()-/$%&*+=<>#@~`^[]{}|";
    const parts = corpus.match(PRETOK) ?? [];
    for (const p of parts) {
      freq.set(p, (freq.get(p) ?? 0) + 1);
      for (const ch of p) seenChars.add(ch);
    }
    this.chars = [...seenChars].sort();

    // --- word representations
    const words: string[] = [];
    const wfreq: number[] = [];
    const wid: WordId[] = [];
    freq.forEach((f, w) => {
      wid.push(words.length);
      words.push(w);
      wfreq.push(f);
    });
    const reps: string[][] = words.map((w) => [...w]);

    // --- pair counts + inverted index
    const key = (a: string, b: string) => a + "" + b;
    const pairCount = new Map<string, number>();
    const pairWords = new Map<string, Set<WordId>>();
    const bump = (k: string, d: number, w: WordId) => {
      pairCount.set(k, (pairCount.get(k) ?? 0) + d);
      if (d > 0) {
        let s = pairWords.get(k);
        if (!s) pairWords.set(k, (s = new Set()));
        s.add(w);
      }
    };
    for (let w = 0; w < reps.length; w++) {
      const t = reps[w], f = wfreq[w];
      for (let i = 0; i + 1 < t.length; i++) bump(key(t[i], t[i + 1]), f, w);
    }

    // --- lazy max heap over pair counts
    const heap: [number, string][] = [];
    const push = (c: number, k: string) => {
      heap.push([c, k]);
      let i = heap.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heap[p][0] >= heap[i][0]) break;
        [heap[p], heap[i]] = [heap[i], heap[p]];
        i = p;
      }
    };
    const pop = (): [number, string] | undefined => {
      if (!heap.length) return undefined;
      const top = heap[0];
      const last = heap.pop()!;
      if (heap.length) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < heap.length && heap[l][0] > heap[m][0]) m = l;
          if (r < heap.length && heap[r][0] > heap[m][0]) m = r;
          if (m === i) break;
          [heap[m], heap[i]] = [heap[i], heap[m]];
          i = m;
        }
      }
      return top;
    };
    pairCount.forEach((c, k) => push(c, k));

    // --- merge loop
    this.merges = [];
    for (let m = 0; m < mergeCount; m++) {
      let best: string | null = null;
      while (heap.length) {
        const [c, k] = pop()!;
        if ((pairCount.get(k) ?? 0) === c && c > 0) { best = k; break; }
      }
      if (best === null) break;
      const sep = best.indexOf("");
      const A = best.slice(0, sep), B = best.slice(sep + 1);
      const M = A + B;
      const affected = pairWords.get(best);
      if (affected) {
        for (const w of [...affected]) {
          const t = reps[w], f = wfreq[w];
          const out: string[] = [];
          let i = 0;
          while (i < t.length) {
            if (i + 1 < t.length && t[i] === A && t[i + 1] === B) {
              if (out.length) {
                const pk = key(out[out.length - 1], A);
                bump(pk, -f, w);
                const nk = key(out[out.length - 1], M);
                bump(nk, f, w);
                push(pairCount.get(nk) ?? 0, nk);
              }
              if (i + 2 < t.length) {
                const pk = key(B, t[i + 2]);
                bump(pk, -f, w);
                const nk = key(M, t[i + 2]);
                bump(nk, f, w);
                push(pairCount.get(nk) ?? 0, nk);
              }
              out.push(M);
              i += 2;
            } else {
              out.push(t[i]);
              i++;
            }
          }
          reps[w] = out;
        }
      }
      pairCount.set(best, 0);
      this.merges.push([A, B]);
      this.mergeRank.set(best, this.merges.length - 1);
      if (onProgress && m % 200 === 0) {
        onProgress(m, mergeCount);
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    // --- final id tables
    this.idToToken = [...SPECIALS, ...this.chars.map((c) => (c === " " ? "·" : c) === "·" && false ? c : c)];
    this.idToToken = [...SPECIALS, ...this.chars];
    for (const [a, b] of this.merges) this.idToToken.push(a + b);
    this.tokenToId = new Map(this.idToToken.map((t, i) => [t, i]));
    this.baseVocab = SPECIALS.length + this.chars.length;
    this.trained = true;
    console.log(`[bpe] trained ${this.merges.length} merges in ${(performance.now() - t0).toFixed(0)}ms, vocab ${this.idToToken.length}`);
  }

  vocabSize(capMerges?: number): number {
    const mc = capMerges === undefined ? this.merges.length : Math.min(capMerges, this.merges.length);
    return this.baseVocab + mc;
  }

  encode(text: string, capMerges?: number): number[] {
    const maxRank = capMerges === undefined ? this.merges.length : Math.min(capMerges, this.merges.length);
    const ids: number[] = [];
    const parts = text.match(PRETOK) ?? [];
    for (const p of parts) {
      let t: string[] = [...p].filter((c) => this.tokenToId.has(c));
      for (;;) {
        let rank = Infinity, ri = -1;
        for (let i = 0; i + 1 < t.length; i++) {
          const r = this.mergeRank.get(t[i] + "" + t[i + 1]);
          if (r !== undefined && r < maxRank && r < rank) { rank = r; ri = i; }
        }
        if (ri < 0) break;
        t = [...t.slice(0, ri), t[ri] + t[ri + 1], ...t.slice(ri + 2)];
      }
      for (const tok of t) {
        const id = this.tokenToId.get(tok);
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  decode(ids: number[]): string {
    let s = "";
    for (const id of ids) {
      const t = this.idToToken[id];
      if (t === undefined) continue;
      s += t;
    }
    return s;
  }

  tokenStr(id: number): string {
    return this.idToToken[id] ?? "?";
  }

  serialize() {
    return { chars: this.chars, merges: this.merges };
  }

  deserialize(data: { chars: string[]; merges: [string, string][] }) {
    this.chars = data.chars;
    this.merges = data.merges;
    this.mergeRank = new Map(data.merges.map(([a, b], i) => [a + "" + b, i]));
    this.idToToken = [...SPECIALS, ...data.chars];
    for (const [a, b] of data.merges) this.idToToken.push(a + b);
    this.tokenToId = new Map(this.idToToken.map((t, i) => [t, i]));
    this.baseVocab = SPECIALS.length + data.chars.length;
    this.trained = true;
  }
}
