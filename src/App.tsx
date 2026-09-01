import { useEffect, useState, useSyncExternalStore } from "react";
import { Cpu } from "lucide-react";
import { engine } from "./engine";
import { Sidebar } from "./ui/Sidebar";
import { TrainView } from "./ui/TrainView";
import { ChatView } from "./ui/ChatView";

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="20" height="20" viewBox="0 0 16 16" className="flex-none">
        <rect width="16" height="16" rx="3.5" fill="#17191d" stroke="#2e323a" strokeWidth="0.75" />
        <rect x="3.25" y="3.25" width="4" height="4" fill="#e8a33d" />
        <rect x="8.75" y="3.25" width="4" height="4" fill="#3a3f47" />
        <rect x="3.25" y="8.75" width="4" height="4" fill="#3a3f47" />
        <rect x="8.75" y="8.75" width="4" height="4" fill="#7fb881" />
      </svg>
      <div className="leading-none">
        <div className="font-bold tracking-[0.18em] text-[13px]">FORGE</div>
        <div className="text-[9.5px] text-dim tracking-wide mt-0.5">in-browser transformer lab</div>
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  const color =
    status === "training" ? "bg-accent animate-pulse" :
    status === "done" ? "bg-good" :
    status === "preparing" ? "bg-accent animate-pulse" :
    status === "error" ? "bg-bad" :
    status === "paused" ? "bg-fog" : "bg-edge2";
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${color}`} />;
}

let didBoot = false;

export default function App() {
  const s = useSyncExternalStore(engine.subscribe, engine.getSnapshot);
  const [tab, setTab] = useState<"train" | "chat">("train");

  useEffect(() => {
    if (didBoot) return;
    didBoot = true;
    engine.init();
  }, []);

  const fmtParams = (n: number) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : String(n));

  return (
    <div className="h-full flex flex-col">
      {/* top bar */}
      <header className="flex-none h-12 border-b border-edge bg-panel flex items-center px-4 gap-4">
        <Logo />
        <div className="seg ml-2">
          <button className={tab === "train" ? "active" : ""} onClick={() => setTab("train")}>Train</button>
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
        </div>
        <div className="flex-1" />
        <div className="hidden md:flex items-center gap-4 font-mono text-[11px] text-fog tabular">
          <span className="flex items-center gap-1.5">
            <StatusDot status={s.status} />
            {s.statusText}
          </span>
          <span className="hidden lg:inline">{s.params ? `${fmtParams(s.params)} params` : "—"}</span>
          <span className="hidden lg:inline">{s.tokPerSec ? `${(s.tokPerSec / 1000).toFixed(1)}k tok/s` : ""}</span>
          <span className={`chip ${s.kernel === "wasm" ? "green" : ""}`}>
            <Cpu size={9} strokeWidth={2.5} />
            {s.kernel === "boot" ? "…" : s.kernel}
          </span>
        </div>
      </header>

      {/* body */}
      <div className="flex-1 min-h-0 flex">
        <Sidebar s={s} />
        <main className="flex-1 min-w-0 flex flex-col min-h-0 bg-ink">
          {tab === "train" ? <TrainView s={s} /> : <ChatView s={s} />}
        </main>
      </div>

      {/* footer */}
      <footer className="flex-none h-7 border-t border-edge bg-panel flex items-center justify-between px-4 text-[10px] font-mono text-dim">
        <span>forge v0.3 · wat-compiled sgemm · bpe tokenizer · adamw + cosine · kv-cached decoding</span>
        <span className="hidden sm:inline">no servers, no apis — every gradient runs in this tab</span>
      </footer>
    </div>
  );
}
