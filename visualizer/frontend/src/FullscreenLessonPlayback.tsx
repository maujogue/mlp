import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
  type WheelEvent,
} from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { LessonReplayManifest, LessonReplayStep } from "./types";
import {
  deriveState,
  epochStartIndices,
  mergedPinSeries,
  softmax,
  type DerivedLessonState,
} from "./lessonUtils";

const CHART_GRID = "#2d3548";
const CHART_AXIS = "#8b95ab";

function signColor(v: number, alpha: number): string {
  if (v >= 0) return `rgba(61, 122, 237, ${alpha})`;
  return `rgba(232, 120, 114, ${alpha})`;
}

function clampMag(w: number, cap: number): number {
  return Math.min(1, Math.abs(w) / Math.max(cap, 1e-9));
}

/** Skip drawing weight lines from ReLU-off sources (a≈0); sigmoid uses a low-activation cutoff. Inputs (col 0) always show. */
function sourceNeuronActiveForEdge(
  col: number,
  j: number,
  d: DerivedLessonState,
  activation: LessonReplayManifest["activation"],
): boolean {
  if (col === 0) return true;
  const a = d.hiddenByLayer[col - 1]?.[j];
  if (a == null) return true;
  if (activation === "sigmoid") return a > 1e-3;
  return a > 1e-9;
}

function stepShortTitle(st: LessonReplayStep): string {
  switch (st.phase) {
    case "init":
      return "Welcome";
    case "forward_input":
      return "Inputs arrive";
    case "forward_layer":
      return `Layer ${(st.layer ?? 0) + 1}: mixing signals`;
    case "activation":
      return "Activation";
    case "loss":
      return "How wrong was the guess?";
    case "backward_layer":
      return "Sending learning backward";
    case "optimizer":
      return "Nudging the weights";
    case "batch_end":
      return "Batch done";
    case "epoch_end":
      return `Epoch ${st.epoch} complete`;
    default:
      return st.phase;
  }
}

function epochRanges(steps: LessonReplayStep[]): { epoch: number; start: number; end: number }[] {
  if (!steps.length) return [];
  const out: { epoch: number; start: number; end: number }[] = [];
  let start = 0;
  let e = steps[0]!.epoch;
  for (let i = 1; i <= steps.length; i++) {
    if (i === steps.length || steps[i]!.epoch !== e) {
      out.push({ epoch: e, start, end: i - 1 });
      if (i < steps.length) {
        start = i;
        e = steps[i]!.epoch;
      }
    }
  }
  return out;
}

type Pin =
  | { kind: "weight"; layer: number; j: number; k: number }
  | { kind: "neuron"; column: number; index: number }
  | { kind: "bias"; layer: number; k: number };

type Props = {
  manifest: LessonReplayManifest;
  steps: LessonReplayStep[];
  onClose: () => void;
};

export default function FullscreenLessonPlayback({ manifest, steps, onClose }: Props) {
  const [idx, setIdx] = useState(0);
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ sx: number; sy: number; px: number; py: number } | null>(null);
  const [pin, setPin] = useState<Pin | null>(null);
  const [showLearnMore, setShowLearnMore] = useState(false);

  const d = useMemo(() => deriveState(steps, idx), [steps, idx]);
  /** Class probabilities for the current step (from replay or softmax(logits)). */
  const outputProbs = useMemo(() => {
    const lg = d.lastLogits;
    const pr = d.lastProbs;
    if (!lg?.length) return null;
    if (pr && pr.length === lg.length) return pr;
    return softmax(lg);
  }, [d.lastLogits, d.lastProbs]);
  const dims = useMemo(
    () => [manifest.input_dim, ...manifest.layer_sizes, manifest.n_classes],
    [manifest],
  );
  const maxW = useMemo(() => {
    let m = 1e-6;
    for (const L of Object.keys(d.W)) {
      const M = d.W[Number(L)];
      if (!M) continue;
      for (const row of M) for (const v of row) m = Math.max(m, Math.abs(v));
    }
    return m;
  }, [d]);

  const ranges = useMemo(() => epochRanges(steps), [steps]);
  const epochStarts = useMemo(() => epochStartIndices(steps), [steps]);
  const n = steps.length;
  const maxIdx = Math.max(0, n - 1);

  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const f = e.deltaY > 0 ? 0.92 : 1.08;
    // Do not zoom out below 1× — keeps the network at least at its base viewBox size.
    setScale((s) => Math.min(4, Math.max(1, s * f)));
  }, []);

  const resetView = useCallback(() => {
    drag.current = null;
    setScale(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const onMouseDownBg = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    drag.current = { sx: e.clientX, sy: e.clientY, px: pan.x, py: pan.y };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!drag.current) return;
    setPan({
      x: drag.current.px + (e.clientX - drag.current.sx),
      y: drag.current.py + (e.clientY - drag.current.sy),
    });
  };
  const onMouseUp = () => {
    drag.current = null;
  };

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (pin) setPin(null);
        else onClose();
      }
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "ArrowRight") setIdx((i) => Math.min(maxIdx, i + 1));
      if (e.key === "ArrowLeft") setIdx((i) => Math.max(0, i - 1));
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [maxIdx, onClose, pin]);

  const colGap = 200;
  const baseX = 100;
  const baseH = 1100;

  const pinModal = useMemo(() => {
    if (!pin) return null;
    const nCols = dims.length;
    let title = "";
    let data: { step: number; value: number | null; grad: number | null }[] = [];
    if (pin.kind === "weight") {
      title = `Link strength from column ${pin.j + 1} to neuron ${pin.k + 1} (layer ${pin.layer + 1})`;
      data = mergedPinSeries(steps, { kind: "weight", layer: pin.layer, j: pin.j, k: pin.k });
    } else if (pin.kind === "bias") {
      title = `Bias for neuron ${pin.k + 1} in layer ${pin.layer + 1}`;
      data = mergedPinSeries(steps, { kind: "bias", layer: pin.layer, k: pin.k });
    } else {
      const col = pin.column;
      title =
        col === 0
          ? `Input feature ${pin.index + 1}`
          : col === nCols - 1
            ? `Output ${pin.index + 1}: probability p and logit z (p = softmax(z))`
            : `Hidden neuron ${pin.index + 1} (layer ${col})`;
      data = mergedPinSeries(steps, {
        kind: "neuron",
        column: col,
        index: pin.index,
        nCols,
      });
    }
    const hasGrad = data.some((row) => row.grad != null);
    const chartMode =
      pin.kind === "neuron" && pin.column === nCols - 1 ? "output_neuron" : "default";
    return { title, data, hasGrad, chartMode };
  }, [pin, steps, dims.length]);

  const minorTickEvery = Math.max(1, Math.ceil(n / 180));
  const tickDenom = maxIdx > 0 ? maxIdx : 1;

  return (
    <div className="flp-root" role="dialog" aria-modal="true" aria-label="Training replay">
      <div className="flp-top">
        <div className="flp-top-text">
          <h2>{stepShortTitle(d.step)}</h2>
          <p className="flp-explain">{d.step.explanation}</p>
          {d.step.math && (
            <div className="flp-more">
              <button type="button" className="flp-btn" onClick={() => setShowLearnMore((v) => !v)}>
                {showLearnMore ? "Hide detail" : "Learn a bit more"}
              </button>
              {showLearnMore && <pre className="flp-math">{d.step.math}</pre>}
            </div>
          )}
        </div>
        <button type="button" className="flp-close" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="flp-note">
        <span className="flp-note-inner">
          {manifest.viz_note ||
            "Scroll to zoom in (up to 4×); you cannot zoom out below the default network size. Drag the background to pan."}
        </span>
        <button type="button" className="flp-btn flp-reset-view" onClick={resetView}>
          Reset view
        </button>
      </div>

      <div className="flp-timeline-wrap">
        <div className="flp-timeline-track" aria-hidden>
          {n > 0 &&
            ranges.map((r) => {
            const w = ((r.end - r.start + 1) / n) * 100;
            const left = (r.start / n) * 100;
            return (
              <div
                key={r.epoch + "-" + r.start}
                className="flp-epoch-band"
                style={{ left: `${left}%`, width: `${w}%` }}
                title={`Epoch ${r.epoch}`}
              />
            );
          })}
          {epochStarts.map((si) => (
            <div
              key={`maj-${si}`}
              className="flp-tick-major"
              style={{ left: `${(si / tickDenom) * 100}%` }}
            />
          ))}
          {Array.from({ length: Math.ceil(n / minorTickEvery) }, (_, t) => t * minorTickEvery)
            .filter((i) => i <= maxIdx)
            .map((i) => (
              <div
                key={`m-${i}`}
                className="flp-tick-minor"
                style={{ left: `${(i / tickDenom) * 100}%` }}
              />
            ))}
        </div>
        <label className="flp-scrub-label">
          Step {idx + 1} / {n}
          <input
            type="range"
            min={0}
            max={maxIdx}
            value={idx}
            onChange={(e) => setIdx(Number(e.target.value))}
            className="flp-scrub"
          />
        </label>
      </div>

      <div
        className="flp-canvas"
        onWheel={onWheel}
        onMouseDown={onMouseDownBg}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <svg
          className="flp-svg"
          viewBox={`0 0 ${baseX + colGap * (dims.length - 1) + 120} ${baseH}`}
        >
          <g transform={`translate(${pan.x} ${pan.y}) scale(${scale})`}>
            <g className="flp-layer-edges">
              {dims.slice(0, -1).map((_, col) => {
                const inC = dims[col]!;
                const outC = dims[col + 1]!;
                const W = d.W[col];
                if (!W) return null;
                const x0 = baseX + col * colGap;
                const x1 = baseX + (col + 1) * colGap;
                const edges: ReactElement[] = [];
                for (let j = 0; j < inC; j++) {
                  if (!sourceNeuronActiveForEdge(col, j, d, manifest.activation)) continue;
                  const y0 = 40 + ((j + 1) * (baseH - 80)) / (inC + 1);
                  for (let k = 0; k < outC; k++) {
                    const y1 = 40 + ((k + 1) * (baseH - 80)) / (outC + 1);
                    const w = W[j]?.[k] ?? 0;
                    const g = d.gradW[col]?.[j]?.[k];
                    const base = clampMag(w, maxW) * 0.85 + 0.12;
                    const sw = 0.35 + base * 2.5;
                    const glow = g != null ? Math.min(6, Math.abs(g) * 0.8) : 0;
                    const sel =
                      pin?.kind === "weight" && pin.layer === col && pin.j === j && pin.k === k
                        ? " flp-sel"
                        : "";
                    edges.push(
                      <line
                        key={`${col}-${j}-${k}`}
                        className={"flp-edge" + sel}
                        x1={x0}
                        y1={y0}
                        x2={x1}
                        y2={y1}
                        stroke={signColor(w, 0.35 + base * 0.5)}
                        strokeWidth={sw + glow * 0.08}
                        onClick={(e) => {
                          e.stopPropagation();
                          setPin({ kind: "weight", layer: col, j, k });
                        }}
                      />,
                    );
                  }
                }
                return <g key={`e-${col}`}>{edges}</g>;
              })}
            </g>
            <g className="flp-layer-nodes">
              {dims.map((count, col) => {
                const x = baseX + col * colGap;
                const nodes = Array.from({ length: count }, (_, r) => {
                  const y = 40 + ((r + 1) * (baseH - 80)) / (count + 1);
                  return { r, y };
                });
                return (
                  <g key={col}>
                    {col === dims.length - 1 ? (
                      <>
                        <text x={x} y={20} textAnchor="middle" className="flp-col-title">
                          Outputs
                        </text>
                        <text x={x} y={34} textAnchor="middle" className="flp-col-sub">
                          p = softmax(z)
                        </text>
                      </>
                    ) : (
                      <text x={x} y={28} textAnchor="middle" className="flp-col-title">
                        {col === 0 ? "Inputs" : `Hidden ${col}`}
                      </text>
                    )}
                    {nodes.map(({ r, y }) => {
                      let fill = "#1c2230";
                      let labelInner: ReactElement | null = null;
                      if (col === 0 && d.input) {
                        labelInner = (
                          <tspan x={x} dy="0.32em" className="flp-nlbl">
                            {d.input[r]?.toFixed(2) ?? ""}
                          </tspan>
                        );
                      } else if (col === dims.length - 1) {
                        const lg = d.lastLogits?.[r];
                        const pv = outputProbs?.[r];
                        fill = r === 1 ? "#2d5c40" : "#2a4070";
                        labelInner = (
                          <>
                            <tspan x={x} dy="-0.2em" className="flp-nlbl-p">
                              {pv != null ? pv.toFixed(3) : ""}
                            </tspan>
                            <tspan x={x} dy="1.05em" className="flp-nlbl-z">
                              {lg != null ? `z ${lg.toFixed(2)}` : ""}
                            </tspan>
                          </>
                        );
                      } else {
                        const hb = d.hiddenByLayer[col - 1];
                        if (hb) {
                          labelInner = (
                            <tspan x={x} dy="0.32em" className="flp-nlbl">
                              {hb[r]?.toFixed(2) ?? ""}
                            </tspan>
                          );
                          fill = "#355a9e";
                        }
                      }
                      const sel =
                        pin?.kind === "neuron" && pin.column === col && pin.index === r ? " flp-sel" : "";
                      return (
                        <g key={r} className="flp-node-wrap">
                          <circle
                            className={"flp-neuron" + sel}
                            cx={x}
                            cy={y}
                            r={13}
                            fill={fill}
                            stroke="#2d3548"
                            strokeWidth={1}
                            pointerEvents="none"
                          />
                          <text
                            x={x}
                            y={y}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            pointerEvents="none"
                          >
                            {labelInner}
                          </text>
                          <circle
                            className="flp-neuron-hit"
                            cx={x}
                            cy={y}
                            r={22}
                            fill="transparent"
                            onClick={(e) => {
                              e.stopPropagation();
                              setPin({ kind: "neuron", column: col, index: r });
                            }}
                          />
                          {col > 0 && (
                            <rect
                              x={x - 52}
                              y={y - 4}
                              width={10}
                              height={8}
                              fill="#5cb87a"
                              opacity={0.6}
                              className={
                                "flp-bias" +
                                (pin?.kind === "bias" && pin.layer === col - 1 && pin.k === r
                                  ? " flp-sel"
                                  : "")
                              }
                              onClick={(e) => {
                                e.stopPropagation();
                                setPin({ kind: "bias", layer: col - 1, k: r });
                              }}
                            />
                          )}
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      </div>

      {pin && pinModal && (
        <div
          className="flp-modal-back"
          role="presentation"
          onClick={() => setPin(null)}
          onKeyDown={(e) => e.key === "Escape" && setPin(null)}
        >
          <div className="flp-modal" role="dialog" onClick={(e) => e.stopPropagation()}>
            <h3>{pinModal.title}</h3>
            <p className="flp-muted">How this value moved during training (each point is a recorded moment).</p>
            <div className="flp-modal-chart">
              {pinModal.data.length === 0 ? (
                <p className="flp-muted" style={{ minHeight: 220, margin: 0, paddingTop: "4rem" }}>
                  No samples for this target appear in the replay log.
                </p>
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={pinModal.data}>
                    <CartesianGrid stroke={CHART_GRID} />
                    <XAxis dataKey="step" stroke={CHART_AXIS} tick={{ fontSize: 10 }} />
                    {pinModal.chartMode === "output_neuron" ? (
                      <>
                        <YAxis
                          yAxisId="p"
                          orientation="left"
                          stroke={CHART_AXIS}
                          width={40}
                          tick={{ fontSize: 9 }}
                          domain={[0, 1]}
                        />
                        <YAxis
                          yAxisId="z"
                          orientation="right"
                          stroke={CHART_AXIS}
                          width={40}
                          tick={{ fontSize: 9 }}
                        />
                      </>
                    ) : (
                      <YAxis stroke={CHART_AXIS} width={44} tick={{ fontSize: 10 }} />
                    )}
                    <Tooltip contentStyle={{ background: "#1c2230", border: "1px solid #2d3548" }} />
                    <Legend />
                    <Line
                      type="monotone"
                      {...(pinModal.chartMode === "output_neuron" ? { yAxisId: "p" } : {})}
                      dataKey="value"
                      name={pinModal.chartMode === "output_neuron" ? "Probability p" : "Value"}
                      stroke="#3d7aed"
                      dot={false}
                      strokeWidth={1.5}
                    />
                    {pinModal.hasGrad && (
                      <Line
                        type="monotone"
                        {...(pinModal.chartMode === "output_neuron" ? { yAxisId: "z" } : {})}
                        dataKey="grad"
                        name={
                          pinModal.chartMode === "output_neuron" ? "Logit z" : "Learning signal"
                        }
                        stroke="#e87872"
                        dot={false}
                        strokeWidth={1.2}
                        connectNulls
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
            <button type="button" className="flp-btn" onClick={() => setPin(null)}>
              Close chart
            </button>
          </div>
        </div>
      )}

      <style>{`
        .flp-root {
          position: fixed;
          inset: 0;
          z-index: 2000;
          background: #0c0e14;
          color: #e8eaed;
          display: flex;
          flex-direction: column;
          font-family: var(--font-sans, system-ui);
        }
        .flp-top {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 1rem;
          padding: 0.75rem 1rem;
          border-bottom: 1px solid #2d3548;
          flex-shrink: 0;
          height: 15rem;
          min-height: 15rem;
          max-height: 15rem;
          overflow-y: auto;
          box-sizing: border-box;
        }
        .flp-top-text {
          flex: 1;
          min-width: 0;
          min-height: 0;
        }
        .flp-top h2 {
          margin: 0 0 0.35rem;
          font-size: 1.15rem;
          line-height: 1.25;
          min-height: 2.875rem;
          max-height: 2.875rem;
          overflow: hidden;
          display: -webkit-box;
          -webkit-box-orient: vertical;
          -webkit-line-clamp: 2;
        }
        .flp-explain {
          margin: 0;
          max-width: 56rem;
          line-height: 1.5;
          color: #c5cad6;
          font-size: 0.95rem;
          height: 4.65rem;
          overflow-y: auto;
          flex-shrink: 0;
        }
        .flp-close {
          flex-shrink: 0;
          background: #2a4a8a;
          border: 1px solid #3d7aed;
          color: #fff;
          padding: 0.4rem 0.9rem;
          border-radius: 8px;
          cursor: pointer;
        }
        .flp-btn {
          background: #1c2230;
          border: 1px solid #2d3548;
          color: #e8eaed;
          padding: 0.35rem 0.75rem;
          border-radius: 6px;
          cursor: pointer;
          margin-top: 0.35rem;
        }
        .flp-more .flp-btn { margin-top: 0; }
        .flp-more { margin-top: 0.25rem; flex-shrink: 0; }
        .flp-math {
          margin: 0.5rem 0 0;
          padding: 0.75rem;
          background: #141821;
          border-radius: 6px;
          font-size: 0.8rem;
          overflow: auto;
          max-height: 5.5rem;
        }
        .flp-note {
          padding: 0.35rem 1rem;
          font-size: 0.8rem;
          color: #8b95ab;
          border-bottom: 1px solid #232a3a;
          flex-shrink: 0;
          height: 3.1rem;
          min-height: 3.1rem;
          max-height: 3.1rem;
          box-sizing: border-box;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          overflow: hidden;
        }
        .flp-note-inner {
          flex: 1;
          min-width: 0;
          display: -webkit-box;
          -webkit-box-orient: vertical;
          -webkit-line-clamp: 2;
          overflow: hidden;
          line-height: 1.35;
        }
        .flp-reset-view {
          flex-shrink: 0;
          margin-top: 0;
          font-size: 0.75rem;
          padding: 0.3rem 0.65rem;
          white-space: nowrap;
        }
        .flp-timeline-wrap {
          padding: 0.5rem 1rem 0.75rem;
          flex-shrink: 0;
          height: 4.85rem;
          min-height: 4.85rem;
          max-height: 4.85rem;
          box-sizing: border-box;
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
        }
        .flp-timeline-track {
          position: relative;
          height: 22px;
          background: #141821;
          border-radius: 4px;
          margin-bottom: 0.35rem;
          overflow: hidden;
        }
        .flp-epoch-band {
          position: absolute;
          top: 0;
          bottom: 0;
          background: rgba(61, 122, 237, 0.12);
          border-right: 1px solid rgba(61, 122, 237, 0.25);
        }
        .flp-tick-major {
          position: absolute;
          top: 0;
          width: 2px;
          margin-left: -1px;
          height: 100%;
          background: rgba(255,255,255,0.35);
          pointer-events: none;
        }
        .flp-tick-minor {
          position: absolute;
          top: 40%;
          width: 1px;
          margin-left: -0.5px;
          height: 60%;
          background: rgba(255,255,255,0.08);
          pointer-events: none;
        }
        .flp-scrub-label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.8rem; color: #8b95ab; }
        .flp-scrub { width: 100%; }
        .flp-canvas {
          flex: 1;
          min-height: 0;
          overflow: hidden;
          cursor: grab;
        }
        .flp-canvas:active { cursor: grabbing; }
        .flp-svg { width: 100%; height: 100%; display: block; }
        .flp-col-title { fill: #8b95ab; font-size: 11px; }
        .flp-col-sub { fill: #6b758a; font-size: 9px; }
        .flp-nlbl-p { fill: #e8eaed; font-size: 7.5px; font-weight: 600; }
        .flp-nlbl-z { fill: #8b95ab; font-size: 6px; }
        .flp-neuron-hit, .flp-edge, .flp-bias { cursor: pointer; }
        .flp-node-wrap:hover .flp-neuron { stroke: #5b90f5; }
        .flp-edge:hover { stroke-opacity: 1; }
        .flp-nlbl { fill: #e8eaed; font-size: 7px; pointer-events: none; }
        .flp-sel { stroke: #fff !important; stroke-width: 2px; }
        .flp-modal-back {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.55);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 2100;
          padding: 1rem;
        }
        .flp-modal {
          background: #141821;
          border: 1px solid #2d3548;
          border-radius: 12px;
          padding: 1rem 1.25rem;
          max-width: 520px;
          width: 100%;
        }
        .flp-modal h3 { margin: 0 0 0.5rem; font-size: 1rem; }
        .flp-muted { color: #8b95ab; font-size: 0.85rem; margin: 0 0 0.75rem; }
        .flp-modal-chart { margin-bottom: 0.5rem; }
        .flp-modal .flp-btn { margin-top: 0.5rem; }
      `}</style>
    </div>
  );
}
