import { useEffect, useRef, useState } from "react";

const ACCENT = "#e8a33d";
const RAW = "rgba(232,163,61,0.22)";
const GRID = "#1e2126";
const TEXT = "#5c6470";

export function LossChart({ series, current }: { series: number[]; current: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [log, setLog] = useState(true);
  const [size, setSize] = useState({ w: 600, h: 240 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(100, r.width), h: Math.max(120, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size.w * dpr;
    cv.height = size.h * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size.w, size.h);

    const padL = 46, padR = 12, padT = 10, padB = 22;
    const W = size.w - padL - padR, H = size.h - padT - padB;
    if (series.length < 2) {
      ctx.font = "11px 'IBM Plex Mono', monospace";
      ctx.fillStyle = TEXT;
      ctx.fillText("loss will plot here once training starts", padL, size.h / 2);
      return;
    }

    let ys = series;
    if (log) ys = ys.map((v) => Math.max(1e-4, Math.log10(Math.max(1e-4, v))));
    let mn = Infinity, mx = -Infinity;
    for (const v of ys) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const pad = (mx - mn) * 0.08 + 1e-6;
    mn -= pad; mx += pad;

    const X = (i: number) => padL + (i / (ys.length - 1)) * W;
    const Y = (v: number) => padT + (1 - (v - mn) / (mx - mn)) * H;

    // grid + y labels
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.strokeStyle = GRID;
    ctx.fillStyle = TEXT;
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const v = mn + (g / 4) * (mx - mn);
      const y = Y(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W, y); ctx.stroke();
      const label = log ? Math.pow(10, v).toFixed(2) : v.toFixed(2);
      ctx.fillText(label, 4, y + 3);
    }
    // x labels
    for (let g = 0; g <= 5; g++) {
      const i = Math.round((g / 5) * (ys.length - 1));
      const x = X(i);
      ctx.fillText(String(i), x - 6, size.h - 6);
    }

    // raw faint
    if (!log || series.length < 3000) {
      ctx.beginPath();
      ctx.strokeStyle = RAW;
      for (let i = 0; i < ys.length; i++) {
        const x = X(i), y = Y(ys[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // EMA
    ctx.beginPath();
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.6;
    let ema = ys[0];
    for (let i = 0; i < ys.length; i++) {
      ema = i === 0 ? ys[i] : ema * 0.96 + ys[i] * 0.04;
      const x = X(i), y = Y(ema);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // current marker
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(X(ys.length - 1), Y(ema), 3, 0, Math.PI * 2);
    ctx.fill();
  }, [series, size, log, current]);

  return (
    <div className="relative h-full" ref={wrapRef}>
      <canvas ref={ref} style={{ width: size.w, height: size.h }} className="block" />
      <button
        onClick={() => setLog((v) => !v)}
        className="absolute top-2 right-2 chip hover:text-paper cursor-pointer bg-panel"
        title="toggle log scale"
      >
        {log ? "log₁₀" : "linear"}
      </button>
    </div>
  );
}
