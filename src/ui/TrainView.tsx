import { useEffect, useRef } from "react";
import { Activity, Gauge, Sigma, Timer, TrendingDown, Waves } from "lucide-react";
import type { EngineState } from "../engine";
import { LossChart } from "./LossChart";

function Stat(props: { icon: React.ReactNode; label: string; value: string; sub?: string }) {
  return (
    <div className="card px-3 py-2.5 min-w-0">
      <div className="flex items-center gap-1.5 text-dim">
        {props.icon}
        <span className="label-micro">{props.label}</span>
      </div>
      <div className="mt-1 font-mono text-[15px] text-paper tabular truncate">{props.value}</div>
      {props.sub && <div className="text-[10px] font-mono text-dim tabular truncate">{props.sub}</div>}
    </div>
  );
}

function fmtTime(ms: number): string {
  if (!isFinite(ms) || ms <= 0) return "—";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

function fmtTok(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return n.toFixed(0);
}

export function TrainView({ s }: { s: EngineState }) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [s.logs.length, s.statusText]);

  const progress = Math.min(1, s.step / Math.max(1, s.totalSteps));
  const eta = s.status === "training" ? fmtTime((s.totalSteps - s.step) * s.msPerStep) : null;

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="p-4 space-y-3 max-w-[1200px]">
        {/* stat strip */}
        <div className="grid grid-cols-3 lg:grid-cols-6 gap-2">
          <Stat icon={<TrendingDown size={11} />} label="Loss" value={isNaN(s.loss) ? "—" : s.loss.toFixed(4)} sub="cross-entropy (ema)" />
          <Stat icon={<Gauge size={11} />} label="Throughput" value={s.tokPerSec ? `${fmtTok(s.tokPerSec)} t/s` : "—"} sub={s.msPerStep ? `${s.msPerStep.toFixed(1)} ms/step` : "not measured"} />
          <Stat icon={<Sigma size={11} />} label="Tokens seen" value={s.tokensSeen.toLocaleString()} sub={`${s.corpusTokens ? (s.tokensSeen / s.corpusTokens).toFixed(1) : "0"} epochs`} />
          <Stat icon={<Waves size={11} />} label="Grad norm" value={s.gnorm ? s.gnorm.toFixed(3) : "—"} sub="clipped @ 1.0" />
          <Stat icon={<Activity size={11} />} label="Learn rate" value={s.lr ? s.lr.toExponential(2) : "—"} sub="warmup + cosine" />
          <Stat icon={<Timer size={11} />} label="ETA" value={eta ?? "—"} sub={`${s.step.toLocaleString()} / ${s.totalSteps.toLocaleString()} steps`} />
        </div>

        {/* chart */}
        <div className="card overflow-hidden">
          <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-edge">
            <span className="label-micro">Loss curve</span>
            <span className="text-[10px] font-mono text-dim">{s.lossSeries.length.toLocaleString()} samples</span>
          </div>
          <div className="h-[240px] dotgrid">
            <LossChart series={s.lossSeries} current={s.loss} />
          </div>
          {/* progress */}
          <div className="border-t border-edge px-3.5 py-2">
            <div className="h-1 rounded-full bg-edge overflow-hidden">
              <div
                className="h-full bg-accent transition-[width] duration-200"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <div className="mt-1.5 flex justify-between text-[10px] font-mono text-dim tabular">
              <span>{s.statusText}</span>
              <span>{(progress * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* console + sample */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div className="card lg:col-span-2 overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-edge">
              <span className="label-micro">Run log</span>
              <span className="text-[10px] font-mono text-dim">stdout, basically</span>
            </div>
            <div ref={logRef} className="h-[210px] overflow-y-auto px-3.5 py-2.5 font-mono text-[11.5px] leading-[1.7] text-fog">
              {s.logs.length === 0 && <div className="text-dim">nothing yet — boot in progress…</div>}
              {s.logs.map((l, i) => (
                <div key={i} className="whitespace-pre-wrap break-words">
                  <span className="text-dim select-none">› </span>
                  <span className={l.startsWith("sample") ? "text-accent" : l.startsWith("done") ? "text-good" : ""}>{l}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-edge">
              <span className="label-micro">Live sample</span>
              <span className="text-[10px] font-mono text-dim tabular">step {s.sampleAt.toLocaleString()}</span>
            </div>
            <div className="flex-1 px-3.5 py-3 font-mono text-[12px] leading-relaxed">
              <div className="text-dim text-[10px] uppercase tracking-wider mb-2">prompt: "hello, how are you?"</div>
              {s.sample ? (
                <div className="text-paper whitespace-pre-wrap">{s.sample}</div>
              ) : (
                <div className="text-dim">every {s.sampleEvery} steps the model answers the same prompt, so you can watch it learn. untrained weights output noise — that's expected.</div>
              )}
            </div>
          </div>
        </div>

        <div className="text-[11px] text-dim leading-relaxed px-1 pb-4">
          Everything you see ran locally: the gradient updates, the tokenizer, the sampling. Close the tab and the weights are gone unless you saved a checkpoint.
        </div>
      </div>
    </div>
  );
}
