import type { LessonReplayManifest, LessonReplayStep } from "./types";

export interface DerivedLessonState {
  input: number[] | null;
  W: Record<number, number[][]>;
  b: Record<number, number[]>;
  hiddenAct: number[] | null;
  /** Last activation vector per hidden layer index (0 .. nHidden-1). */
  hiddenByLayer: Record<number, number[]>;
  lastLogits: number[] | null;
  lastProbs: number[] | null;
  gradW: Record<number, number[][]>;
  gradB: Record<number, number[]>;
  edgeContrib: Record<number, number[][] | undefined>;
  step: LessonReplayStep;
}

export function deriveState(steps: LessonReplayStep[], i: number): DerivedLessonState {
  let input: number[] | null = null;
  const W: Record<number, number[][]> = {};
  const b: Record<number, number[]> = {};
  let hiddenAct: number[] | null = null;
  const hiddenByLayer: Record<number, number[]> = {};
  let lastLogits: number[] | null = null;
  let lastProbs: number[] | null = null;
  const gradW: Record<number, number[][]> = {};
  const gradB: Record<number, number[]> = {};
  const edgeContrib: Record<number, number[][] | undefined> = {};
  const n = Math.max(0, Math.min(i, steps.length - 1));
  for (let s = 0; s <= n; s++) {
    const st = steps[s];
    if (st.phase === "forward_input" && st.a_in) input = st.a_in;
    if (st.phase === "forward_layer" && st.layer != null && st.W) {
      W[st.layer] = st.W;
      if (st.b) b[st.layer] = st.b;
      if (st.edge_contributions) edgeContrib[st.layer] = st.edge_contributions;
    }
    if (st.phase === "activation" && st.a_out) {
      hiddenAct = st.a_out;
      if (st.layer != null) hiddenByLayer[st.layer] = st.a_out;
    }
    if (st.phase === "loss") {
      if (st.logits) lastLogits = st.logits;
      if (st.probs) lastProbs = st.probs;
    }
    if (st.phase === "backward_layer" && st.layer != null) {
      if (st.dL_dW) gradW[st.layer] = st.dL_dW;
      if (st.dL_db) gradB[st.layer] = st.dL_db;
    }
  }
  return {
    input,
    W,
    b,
    hiddenAct,
    hiddenByLayer,
    lastLogits,
    lastProbs,
    gradW,
    gradB,
    edgeContrib,
    step: steps[n],
  };
}

export function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

export function softmax(logits: number[]): number[] {
  const shift = Math.max(...logits);
  const ex = logits.map((t) => Math.exp(t - shift));
  const s = ex.reduce((a, c) => a + c, 0);
  return ex.map((e) => e / s);
}

function matVec(x: number[], W: number[][]): number[] {
  const outD = W[0]?.length ?? 0;
  const res = new Array(outD).fill(0);
  for (let j = 0; j < x.length; j++) {
    const row = W[j];
    if (!row) continue;
    for (let i = 0; i < outD; i++) res[i] += x[j] * row[i]!;
  }
  return res;
}

function addVec(a: number[], bvec: number[]): number[] {
  return a.map((v, i) => v + (bvec[i] ?? 0));
}

export function forwardFromState(
  x: number[],
  manifest: LessonReplayManifest,
  d: DerivedLessonState,
): { logits: number[]; probs: number[] } | null {
  const nHidden = manifest.layer_sizes.length;
  if (nHidden !== 1) return null;
  const W0 = d.W[0];
  const b0 = d.b[0];
  const W1 = d.W[1];
  const b1 = d.b[1];
  if (!W0 || !b0 || !W1 || !b1) return null;
  let h = addVec(matVec(x, W0), b0);
  if (manifest.activation === "sigmoid") h = h.map(sigmoid);
  else h = h.map((t) => Math.max(0, t));
  const logits = addVec(matVec(h, W1), b1);
  return { logits, probs: softmax(logits) };
}

export function lossCurve(steps: LessonReplayStep[]): { step: number; loss: number }[] {
  const pts: { step: number; loss: number }[] = [];
  for (const st of steps) {
    if (st.phase === "loss" && st.loss_contribution != null) {
      pts.push({ step: st.step_index, loss: st.loss_contribution });
    }
  }
  return pts;
}

export function weightHistory(
  steps: LessonReplayStep[],
  layer: number,
  j: number,
  k: number,
): number[] {
  const out: number[] = [];
  for (const st of steps) {
    if (st.phase === "forward_layer" && st.layer === layer && st.W?.[j]?.[k] != null) {
      out.push(st.W[j]![k]!);
    }
  }
  return out;
}

export function gradWeightHistory(
  steps: LessonReplayStep[],
  layer: number,
  j: number,
  k: number,
): number[] {
  const out: number[] = [];
  for (const st of steps) {
    if (st.phase === "backward_layer" && st.layer === layer && st.dL_dW?.[j]?.[k] != null) {
      out.push(st.dL_dW[j]![k]!);
    }
  }
  return out;
}

export function biasHistory(
  steps: LessonReplayStep[],
  layer: number,
  k: number,
): number[] {
  const out: number[] = [];
  for (const st of steps) {
    if (st.phase === "forward_layer" && st.layer === layer && st.b?.[k] != null) {
      out.push(st.b[k]!);
    }
  }
  return out;
}

export function gradBiasHistory(steps: LessonReplayStep[], layer: number, k: number): number[] {
  const out: number[] = [];
  for (const st of steps) {
    if (st.phase === "backward_layer" && st.layer === layer && st.dL_db?.[k] != null) {
      out.push(st.dL_db[k]!);
    }
  }
  return out;
}

export function neuronValueHistory(
  steps: LessonReplayStep[],
  column: number,
  neuronIndex: number,
  nCols: number,
): number[] {
  const out: number[] = [];
  if (column === 0) {
    for (const st of steps) {
      if (st.phase === "forward_input" && st.a_in?.[neuronIndex] != null) {
        out.push(st.a_in[neuronIndex]!);
      }
    }
    return out;
  }
  if (column === nCols - 1) {
    for (const st of steps) {
      if (st.phase === "loss" && st.logits?.[neuronIndex] != null) {
        out.push(st.logits[neuronIndex]!);
      }
    }
    return out;
  }
  const hiddenLayer = column - 1;
  for (const st of steps) {
    if (
      st.phase === "activation" &&
      st.layer === hiddenLayer &&
      st.a_out?.[neuronIndex] != null
    ) {
      out.push(st.a_out[neuronIndex]!);
    }
  }
  return out;
}

/** First step index where each epoch starts (1-based epoch labels). */
export function epochStartIndices(steps: LessonReplayStep[]): number[] {
  const starts: number[] = [0];
  for (let i = 1; i < steps.length; i++) {
    if (steps[i]!.epoch !== steps[i - 1]!.epoch) starts.push(i);
  }
  return starts;
}

export type PinSeriesKind =
  | { kind: "weight"; layer: number; j: number; k: number }
  | { kind: "bias"; layer: number; k: number }
  | { kind: "neuron"; column: number; index: number; nCols: number };

/** Merge value + gradient (when logged) by global step_index for charts. */
export function mergedPinSeries(
  steps: LessonReplayStep[],
  pin: PinSeriesKind,
): { step: number; value: number | null; grad: number | null }[] {
  const m = new Map<number, { value?: number; grad?: number }>();
  if (pin.kind === "weight") {
    for (const st of steps) {
      if (st.phase === "forward_layer" && st.layer === pin.layer && st.W?.[pin.j]?.[pin.k] != null) {
        const cur = m.get(st.step_index) ?? {};
        cur.value = st.W[pin.j]![pin.k]!;
        m.set(st.step_index, cur);
      }
      if (
        st.phase === "backward_layer" &&
        st.layer === pin.layer &&
        st.dL_dW?.[pin.j]?.[pin.k] != null
      ) {
        const cur = m.get(st.step_index) ?? {};
        cur.grad = st.dL_dW[pin.j]![pin.k]!;
        m.set(st.step_index, cur);
      }
    }
  } else if (pin.kind === "bias") {
    for (const st of steps) {
      if (st.phase === "forward_layer" && st.layer === pin.layer && st.b?.[pin.k] != null) {
        const cur = m.get(st.step_index) ?? {};
        cur.value = st.b[pin.k]!;
        m.set(st.step_index, cur);
      }
      if (st.phase === "backward_layer" && st.layer === pin.layer && st.dL_db?.[pin.k] != null) {
        const cur = m.get(st.step_index) ?? {};
        cur.grad = st.dL_db[pin.k]!;
        m.set(st.step_index, cur);
      }
    }
  } else {
    const { column: col, index: idx, nCols } = pin;
    if (col === 0) {
      for (const st of steps) {
        if (st.phase === "forward_input" && st.a_in?.[idx] != null) {
          m.set(st.step_index, { value: st.a_in[idx]! });
        }
      }
    } else if (col === nCols - 1) {
      for (const st of steps) {
        if (st.phase === "loss" && st.logits?.[idx] != null) {
          const lg = st.logits[idx]!;
          const pr =
            st.probs?.[idx] ??
            (st.logits.length > 0 ? softmax(st.logits)[idx]! : undefined);
          const cur = m.get(st.step_index) ?? {};
          cur.value = pr ?? lg;
          cur.grad = lg;
          m.set(st.step_index, cur);
        }
      }
    } else {
      const hl = col - 1;
      for (const st of steps) {
        if (st.phase === "activation" && st.layer === hl && st.a_out?.[idx] != null) {
          m.set(st.step_index, { value: st.a_out[idx]! });
        }
      }
    }
  }
  return [...m.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, o]) => ({
      step,
      value: o.value ?? null,
      grad: o.grad ?? null,
    }));
}

export function flattenParamIndex(
  manifest: LessonReplayManifest,
  layer: number,
  j: number,
  k: number,
  kind: "W" | "b",
): number {
  const dims = [manifest.input_dim, ...manifest.layer_sizes, manifest.n_classes];
  let idx = 0;
  for (let L = 0; L < dims.length - 1; L++) {
    const inD = dims[L]!;
    const outD = dims[L + 1]!;
    const wSize = inD * outD;
    if (L === layer && kind === "W") {
      return idx + j * outD + k;
    }
    idx += wSize;
    if (L === layer && kind === "b") {
      return idx + k;
    }
    idx += outD;
  }
  return -1;
}

export function totalParamCount(manifest: LessonReplayManifest): number {
  const dims = [manifest.input_dim, ...manifest.layer_sizes, manifest.n_classes];
  let n = 0;
  for (let L = 0; L < dims.length - 1; L++) {
    const inD = dims[L]!;
    const outD = dims[L + 1]!;
    n += inD * outD + outD;
  }
  return n;
}
