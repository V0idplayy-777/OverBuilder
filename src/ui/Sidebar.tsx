import { AlertTriangle, Database, Download, Pause, Play, RotateCcw, SlidersHorizontal, Upload } from "lucide-react";
import { DATASETS, engine, type EngineState } from "../engine";
import { MODELS } from "../nn/configs";
import { useRef } from "react";

function NumField(props: {
  label: string; value: number; min: number; max?: number; step?: number;
  onCommit: (v: number) => void; format?: (v: number) => string; parse?: (s: string) => number;
  suffix?: string; disabled?: boolean;
}) {
  const fmt = props.format ?? ((v: number) => String(v));
  const parse = props.parse ?? ((s: string) => parseFloat(s));
  return (
    <label className="block">
      <div className="flex justify-between items-baseline mb-1">
        <span className="label-micro">{props.label}</span>
        {props.suffix && <span className="text-[10px] text-dim font-mono">{props.suffix}</span>}
      </div>
      <input
        className="input-num tabular"
        type="text"
        inputMode="decimal"
        defaultValue={fmt(props.value)}
        key={fmt(props.value)}
        disabled={props.disabled}
        onBlur={(e) => {
          const v = parse(e.target.value.replace(/[^0-9eE.\-+]/g, ""));
          if (isFinite(v)) props.onCommit(Math.max(props.min, props.max ? Math.min(props.max, v) : v));
          else e.target.value = fmt(props.value);
        }}
        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
      />
    </label>
  );
}

export function Sidebar({ s }: { s: EngineState }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const busy = s.status === "preparing" || !s.modelReady;
  const training = s.status === "training";

  const download = () => {
    const blob = engine.exportCheckpoint();
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `forge-${s.modelId}-step${s.step}.forge`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  };

  return (
    <aside className="w-[300px] flex-none border-r border-edge bg-panel flex flex-col min-h-0">
      <div className="flex-1 overflow-y-auto">
        {/* ---------- model ---------- */}
        <div className="px-4 pt-4 pb-2 flex items-center justify-between">
          <span className="label-micro">Model</span>
          <span className="text-[10px] font-mono text-dim">tied embeddings</span>
        </div>
        <div className="px-3 pb-3 space-y-1.5">
          {MODELS.map((m) => {
            const active = m.id === s.modelId;
            const mem = m.memGB;
            return (
              <button
                key={m.id}
                className={`model-row ${active ? "active" : ""}`}
                disabled={busy || training || s.generating}
                onClick={() => engine.selectModel(m.id)}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-semibold text-paper">{m.name}</span>
                  <span className="flex items-center gap-1.5">
                    {m.heavy && (
                      <span className="chip amber"><AlertTriangle size={9} strokeWidth={2.5} /> heavy</span>
                    )}
                    <span className="text-[11px] font-mono text-fog tabular">{m.tier.replace("~", "")}</span>
                  </span>
                </div>
                <div className="mt-0.5 text-[11px] font-mono text-dim tabular">
                  d {m.d} · {m.layers}L · {m.heads}h · ctx {m.ctx} · ~{mem.toFixed(2)} gb
                </div>
                {active && (
                  <div className="mt-1.5 text-[11px] leading-snug text-fog">{m.blurb}</div>
                )}
              </button>
            );
          })}
        </div>

        {/* ---------- dataset ---------- */}
        <div className="px-4 pt-2 pb-2 flex items-center gap-1.5">
          <Database size={11} className="text-dim" />
          <span className="label-micro">Dataset</span>
        </div>
        <div className="px-3 pb-3 space-y-1.5">
          {DATASETS().map((d) => {
            const active = d.id === s.datasetId;
            return (
              <button
                key={d.id}
                className={`model-row ${active ? "active" : ""}`}
                disabled={busy || training || s.generating}
                onClick={() => engine.selectDataset(d.id)}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-medium text-paper">{d.name}</span>
                  <span className="text-[11px] font-mono text-dim tabular">{(d.text.length / 1024).toFixed(0)} kb</span>
                </div>
                {active && <div className="mt-1 text-[11px] leading-snug text-fog">{d.blurb}</div>}
              </button>
            );
          })}
          <div className="pt-1 px-1 text-[11px] font-mono text-dim tabular leading-relaxed">
            vocab {s.vocabSize ? s.vocabSize.toLocaleString() : "—"} · {s.corpusTokens ? s.corpusTokens.toLocaleString() : "—"} tokens
          </div>
        </div>

        {/* ---------- optimizer ---------- */}
        <div className="px-4 pt-2 pb-2 flex items-center gap-1.5">
          <SlidersHorizontal size={11} className="text-dim" />
          <span className="label-micro">Optimizer</span>
        </div>
        <div className="px-4 pb-4 grid grid-cols-2 gap-2">
          <NumField label="Steps" value={s.totalSteps} min={10} max={200000}
            onCommit={(v) => engine.update({ totalSteps: Math.round(v) })} />
          <NumField label="Peak LR" value={s.lrPeak} min={1e-5} max={0.1}
            format={(v) => v.toExponential(1)} parse={(x) => parseFloat(x)}
            onCommit={(v) => engine.update({ lrPeak: v })} />
          <NumField label="Context" value={s.trainCtx} min={16} max={SPEC_CTX(s.modelId)} step={4} suffix={`max ${SPEC_CTX(s.modelId)}`}
            onCommit={(v) => engine.update({ trainCtx: Math.round(v / 4) * 4 })} />
          <NumField label="Sample every" value={s.sampleEvery} min={10} max={5000}
            onCommit={(v) => engine.update({ sampleEvery: Math.round(v) })} />
        </div>
      </div>

      {/* ---------- actions ---------- */}
      <div className="flex-none border-t border-edge p-3 space-y-2 bg-panel">
        {training ? (
          <button className="btn btn-ghost w-full justify-center" onClick={() => engine.pause()}>
            <Pause size={14} /> Pause at step {s.step.toLocaleString()}
          </button>
        ) : (
          <button
            className="btn btn-primary w-full justify-center"
            disabled={busy || s.generating}
            onClick={() => engine.start()}
          >
            <Play size={14} />
            {s.status === "paused" && s.step < s.totalSteps
              ? `Resume · step ${s.step.toLocaleString()}`
              : s.step > 0 && s.step < s.totalSteps
                ? `Continue · step ${s.step.toLocaleString()}`
                : `Train ${s.totalSteps.toLocaleString()} steps`}
          </button>
        )}
        <div className="flex gap-2">
          <button className="btn btn-ghost flex-1 justify-center" disabled={busy || training} onClick={() => engine.resetTraining()}>
            <RotateCcw size={13} /> Reset
          </button>
          <button className="btn btn-ghost flex-1 justify-center" disabled={busy || !s.trained} onClick={download}>
            <Download size={13} /> Save
          </button>
          <button className="btn btn-ghost flex-1 justify-center" disabled={busy || training} onClick={() => fileRef.current?.click()}>
            <Upload size={13} /> Load
          </button>
          <input
            ref={fileRef} type="file" accept=".forge" className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) engine.importCheckpoint(f);
              e.target.value = "";
            }}
          />
        </div>
        <div className="text-[10px] leading-snug text-dim px-0.5">
          Checkpoints are plain binary files — weights, tokenizer merges and config, nothing else. Loading one replaces the current model.
        </div>
      </div>
    </aside>
  );
}

function SPEC_CTX(id: string): number {
  return MODELS.find((m) => m.id === id)?.ctx ?? 128;
}
