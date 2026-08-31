import { useEffect, useState, useSyncExternalStore } from "react";
import { engine } from "./engine";
import { Sidebar } from "./ui/Sidebar";
import { TrainView } from "./ui/TrainView";
import { ChatView } from "./ui/ChatView";

let initPromise: Promise<void> | null = null;

function initEngine() {
  if (!initPromise) {
    initPromise = engine.init();
  }
  return initPromise;
}

export default function App() {
  const state = useSyncExternalStore(
    engine.subscribe,
    engine.getSnapshot,
    engine.getSnapshot
  );

  const [tab, setTab] = useState<"train" | "chat">("train");

  useEffect(() => {
    initEngine().catch((error) => {
      console.error(error);
    });
  }, []);

  const statusLabel =
    state.status === "preparing"
      ? state.statusText
      : state.status === "training"
        ? `training · step ${state.step.toLocaleString()} / ${state.totalSteps.toLocaleString()}`
        : state.status === "done"
          ? "training complete"
          : state.status === "paused"
            ? "paused"
            : state.status === "error"
              ? "error"
              : state.statusText;

  return (
    <div className="min-h-screen h-screen bg-[#0b0d0f] text-[#e8eaed] flex flex-col overflow-hidden">
      <header className="h-12 flex-none border-b border-[#1e2126] bg-[#111315] flex items-center justify-between px-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-7 h-7 rounded-md bg-[#e8a33d] flex items-center justify-center text-[#0b0d0f] font-black text-sm">
            F
          </div>

          <div className="flex items-center gap-2 min-w-0">
            <span className="font-semibold tracking-tight text-[14px]">
              Forge
            </span>
            <span className="text-[#5c6470] text-[12px]">/</span>
            <span className="font-mono text-[11px] text-[#8b929c] truncate">
              local language model lab
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-[10px] font-mono text-[#8b929c]">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                state.kernel === "wasm"
                  ? "bg-emerald-400"
                  : state.kernel === "js"
                    ? "bg-amber-400"
                    : "bg-[#5c6470]"
              }`}
            />
            {state.kernel === "wasm"
              ? "wasm"
              : state.kernel === "js"
                ? "javascript"
                : "booting"}
          </div>

          <div className="text-[10px] font-mono text-[#5c6470]">
            {statusLabel}
          </div>
        </div>
      </header>

      <div className="h-10 flex-none border-b border-[#1e2126] bg-[#0f1113] flex items-center px-3">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setTab("train")}
            className={`px-3 py-1.5 rounded-md text-[11px] font-mono transition-colors ${
              tab === "train"
                ? "bg-[#1e2126] text-[#e8eaed]"
                : "text-[#5c6470] hover:text-[#aeb4bc]"
            }`}
          >
            train
          </button>

          <button
            type="button"
            onClick={() => setTab("chat")}
            className={`px-3 py-1.5 rounded-md text-[11px] font-mono transition-colors ${
              tab === "chat"
                ? "bg-[#1e2126] text-[#e8eaed]"
                : "text-[#5c6470] hover:text-[#aeb4bc]"
            }`}
          >
            chat
          </button>
        </div>

        <div className="ml-auto flex items-center gap-4 text-[10px] font-mono text-[#5c6470]">
          <span>
            {state.modelId} · {state.params ? state.params.toLocaleString() : "—"} params
          </span>
          <span className="hidden md:inline">
            {state.vocabSize ? `${state.vocabSize.toLocaleString()} vocab` : "vocab —"}
          </span>
        </div>
      </div>

      <main className="flex-1 min-h-0 flex">
        <Sidebar s={state} />

        <section className="flex-1 min-w-0 min-h-0 flex">
          {tab === "train" ? <TrainView s={state} /> : <ChatView s={state} />}
        </section>
      </main>
    </div>
  );
}
