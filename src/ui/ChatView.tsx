import { useEffect, useRef, useState } from "react";
import { ArrowUp, Settings2, Square, Trash2, Zap } from "lucide-react";
import { engine, type EngineState } from "../engine";

const SUGGESTIONS = ["hello!", "what is your name?", "tell me a joke", "what can you do?", "what is 3 plus 7?", "what is wasm?"];

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="block">
      <div className="flex justify-between text-[10px] font-mono text-fog mb-1">
        <span className="uppercase tracking-wider">{props.label}</span>
        <span className="tabular">{(props.fmt ?? ((v) => v.toFixed(2)))(props.value)}</span>
      </div>
      <input
        type="range" className="w-full" min={props.min} max={props.max} step={props.step}
        value={props.value} onChange={(e) => props.onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function ChatView({ s }: { s: EngineState }) {
  const [text, setText] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [s.chats.length, s.chats[s.chats.length - 1]?.text]);

  const send = (t?: string) => {
    const msg = (t ?? text).trim();
    if (!msg || s.generating || !s.trained || s.status === "preparing") return;
    setText("");
    engine.chat(msg);
  };

  const ready = s.trained && s.step > 0;

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* header */}
      <div className="flex-none border-b border-edge px-4 py-2 flex items-center justify-between bg-panel">
        <div className="flex items-center gap-3">
          <span className="label-micro">Chat</span>
          {ready ? (
            <span className="chip green">{s.modelId} · step {s.step.toLocaleString()}</span>
          ) : (
            <span className="chip">untrained</span>
          )}
          {s.generating && <span className="chip amber">generating</span>}
        </div>
        <div className="flex items-center gap-1.5">
          <button className="btn btn-ghost !px-2.5 !py-1.5" onClick={() => setShowSettings((v) => !v)} title="sampling settings">
            <Settings2 size={14} />
          </button>
          <button className="btn btn-ghost !px-2.5 !py-1.5" onClick={() => engine.clearChat()} disabled={s.chats.length === 0} title="clear conversation">
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {showSettings && (
        <div className="flex-none border-b border-edge bg-panel px-4 py-3 grid grid-cols-2 md:grid-cols-5 gap-x-5 gap-y-2.5">
          <Slider label="temperature" value={s.temp} min={0.1} max={1.5} step={0.05} onChange={(v) => engine.update({ temp: v })} />
          <Slider label="top-k" value={s.topK} min={1} max={100} step={1} fmt={(v) => String(Math.round(v))} onChange={(v) => engine.update({ topK: Math.round(v) })} />
          <Slider label="top-p" value={s.topP} min={0.5} max={1} step={0.01} onChange={(v) => engine.update({ topP: v })} />
          <Slider label="rep. penalty" value={s.repPen} min={1} max={2} step={0.05} onChange={(v) => engine.update({ repPen: v })} />
          <Slider label="max tokens" value={s.maxNew} min={8} max={96} step={4} fmt={(v) => String(Math.round(v))} onChange={(v) => engine.update({ maxNew: Math.round(v) })} />
        </div>
      )}

      {/* messages */}
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-5 space-y-4">
          {!ready && (
            <div className="card p-5">
              <div className="text-[15px] font-semibold text-paper">No trained weights yet</div>
              <p className="mt-1.5 text-[13px] text-fog leading-relaxed">
                This chat speaks with the model from the train tab — nothing else. Give it a body first:
                quick-train the smallest one (about a minute), or pick any size and train it yourself.
              </p>
              <div className="mt-3.5 flex gap-2">
                <button className="btn btn-primary" onClick={() => engine.quickTrain()}>
                  <Zap size={14} /> Quick-train Nano
                </button>
              </div>
              {s.status === "training" && (
                <div className="mt-3 text-[11px] font-mono text-accent tabular">
                  training… step {s.step.toLocaleString()} / {s.totalSteps.toLocaleString()} — loss {s.loss.toFixed(3)}
                </div>
              )}
            </div>
          )}

          {ready && s.chats.length === 0 && (
            <div className="pt-8">
              <div className="text-center text-[13px] text-dim">Say something. It's a {s.modelId} model — expectations calibrated accordingly.</div>
              <div className="mt-4 flex flex-wrap justify-center gap-1.5">
                {SUGGESTIONS.map((q) => (
                  <button key={q} className="btn btn-ghost !py-1 !px-2.5 !text-[12px]" onClick={() => send(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {s.chats.map((m, i) => (
            <div key={i} className={`msg-row ${m.role === "u" ? "flex-row-reverse" : ""}`}>
              <div className={`msg-tag ${m.role}`}>{m.role === "u" ? "you" : "ai"}</div>
              <div
                className={`max-w-[80%] rounded-lg px-3.5 py-2.5 text-[13.5px] leading-relaxed whitespace-pre-wrap break-words ${
                  m.role === "u"
                    ? "bg-panel2 border border-edge2"
                    : `bg-panel border border-edge ${s.generating && i === s.chats.length - 1 ? "caret" : ""}`
                }`}
              >
                {m.text || (s.generating && i === s.chats.length - 1 ? "" : "…")}
              </div>
            </div>
          ))}
          <div className="h-2" />
        </div>
      </div>

      {/* composer */}
      <div className="flex-none border-t border-edge bg-panel px-4 py-3">
        <div className="max-w-2xl mx-auto">
          <div className="flex gap-2 items-center">
            <input
              className="input-text !py-2.5 !text-[13.5px] !font-sans flex-1"
              placeholder={ready ? "message the model…" : "train a model first"}
              value={text}
              disabled={!ready || s.generating}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            />
            {s.generating ? (
              <button className="btn btn-ghost !py-2.5" onClick={() => engine.stopChat()} title="stop">
                <Square size={14} />
              </button>
            ) : (
              <button className="btn btn-primary !py-2.5" onClick={() => send()} disabled={!ready || !text.trim()} title="send">
                <ArrowUp size={14} />
              </button>
            )}
          </div>
          <div className="mt-1.5 flex justify-between text-[10px] text-dim font-mono">
            <span>enter to send · replies come from your trained weights, streamed token by token</span>
            <span className="tabular">ctx {s.trained ? "" : "—"}{ready ? Math.min(s.maxNew, 96) : ""}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
