import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FullscreenLessonPlayback from "./FullscreenLessonPlayback.tsx";
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
import { DataManagementPanel } from "./components/DataManagementPanel";
import { DatasetCsvPickRow } from "./components/DatasetCsvPickRow";
import { GridSearchPanel } from "./components/GridSearchPanel";
import type {
  DatasetListResponse,
  EvaluateTestResponse,
  LessonReplayManifest,
  LessonReplayStep,
  RunDetailResponse,
  RunListItem,
  RunListResponse,
  SseEvent,
  TestEvalResult,
  TrainingHistoryJson,
} from "./types";

const COLORS = ["#3d7aed", "#e87872", "#5cb87a", "#e4b04a", "#b279a2", "#ff9da6"];

/** Hidden <select> option: apply preset to hidden layers string via onChange */
const LIVE_LAYER_PRESETS: { label: string; value: string }[] = [
  { label: "24, 24, 12", value: "24,24,12" },
  { label: "24, 24", value: "24,24" },
  { label: "32, 16", value: "32,16" },
  { label: "20, 10", value: "20,10" },
  { label: "16, 8", value: "16,8" },
];

const LIVE_VAL_RATIO_PRESETS: { label: string; value: number }[] = [
  { label: "10%", value: 0.1 },
  { label: "15%", value: 0.15 },
  { label: "20%", value: 0.2 },
  { label: "25%", value: 0.25 },
  { label: "30%", value: 0.3 },
];

const LIVE_BATCH_PRESETS: { label: string; value: number }[] = [
  { label: "Full batch (0)", value: 0 },
  { label: "16", value: 16 },
  { label: "32", value: 32 },
  { label: "64", value: 64 },
  { label: "128", value: 128 },
];

const LIVE_SEED_PRESETS: { label: string; value: number }[] = [
  { label: "42", value: 42 },
  { label: "0", value: 0 },
  { label: "1", value: 1 },
  { label: "123", value: 123 },
];

const CHART_GRID = "#2d3548";
const CHART_AXIS = "#8b95ab";
const CHART_TOOLTIP = { background: "#1c2230", border: "1px solid #2d3548" };

type RunSort =
  | "val_loss_asc"
  | "val_loss_desc"
  | "train_loss_asc"
  | "train_loss_desc"
  | "test_bce_asc"
  | "test_bce_desc"
  | "newest"
  | "path_asc";

type DatePreset = "any" | "24h" | "7d";

type ExperimentFilters = {
  trainPath: string;
  layers: string;
  epochs: string;
  lr: string;
  seed: string;
  batch: string;
  optimizer: string;
  patience: string;
};

function uniqSortedStrings(values: (string | null | undefined)[]): string[] {
  const s = new Set<string>();
  for (const v of values) {
    if (v != null && v !== "") s.add(v);
  }
  return [...s].sort((a, b) => a.localeCompare(b));
}

function uniqSortedNumberKeys(values: (number | null | undefined)[]): string[] {
  const s = new Set<string>();
  for (const v of values) {
    if (v != null) s.add(String(v));
  }
  return [...s].sort((a, b) => Number(a) - Number(b));
}

function runMatchesExperiment(r: RunListItem, f: ExperimentFilters): boolean {
  if (f.trainPath && r.config_train_path !== f.trainPath) return false;
  if (f.layers && r.config_layers_str !== f.layers) return false;
  if (f.epochs && String(r.config_epochs ?? "") !== f.epochs) return false;
  if (f.lr && String(r.config_learning_rate ?? "") !== f.lr) return false;
  if (f.seed && String(r.config_seed ?? "") !== f.seed) return false;
  if (f.batch && String(r.config_batch_size ?? "") !== f.batch) return false;
  if (f.optimizer && r.config_optimizer !== f.optimizer) return false;
  if (f.patience && String(r.config_patience ?? "") !== f.patience) return false;
  return true;
}

function testLossForSort(r: RunListItem, testResults: Record<string, TestEvalResult>): number {
  const t = testResults[r.id];
  if (t && "loss" in t) return t.loss;
  return Number.NaN;
}

function formatTestCell(
  r: RunListItem,
  testResults: Record<string, TestEvalResult>,
): { cell: string; title: string } {
  const t = testResults[r.id];
  if (!t) return { cell: "—", title: "" };
  if ("error" in t) return { cell: "err", title: t.error };
  return { cell: t.loss.toFixed(4), title: `acc ${t.accuracy.toFixed(4)} · F1 ${t.f1.toFixed(4)}` };
}

function maxEpochs(h: TrainingHistoryJson): number {
  return Math.max(h.history_train_loss.length, h.history_val_loss.length, 1);
}

function buildSeries(
  h: TrainingHistoryJson,
  cap: number,
): { epoch: number; train_loss: number; val_loss?: number; train_acc?: number; val_acc?: number }[] {
  const n = Math.min(cap, maxEpochs(h));
  const rows: {
    epoch: number;
    train_loss: number;
    val_loss?: number;
    train_acc?: number;
    val_acc?: number;
  }[] = [];
  for (let i = 0; i < n; i++) {
    const row: (typeof rows)[0] = {
      epoch: i + 1,
      train_loss: h.history_train_loss[i] ?? 0,
    };
    if (h.history_val_loss[i] !== undefined) {
      row.val_loss = h.history_val_loss[i];
    }
    if (h.history_train_acc[i] !== undefined) {
      row.train_acc = h.history_train_acc[i];
    }
    if (h.history_val_acc[i] !== undefined) {
      row.val_acc = h.history_val_acc[i];
    }
    rows.push(row);
  }
  return rows;
}

function annotationForEpoch(h: TrainingHistoryJson, epochIdx: number): string | null {
  const i = epochIdx - 1;
  if (i < 1) return null;
  const tl = h.history_train_loss;
  const vl = h.history_val_loss;
  if (!vl.length || i >= tl.length || i >= vl.length) return null;
  if (tl[i] < tl[i - 1] && vl[i] > vl[i - 1]) {
    return "Train loss decreased while validation loss increased — a common sign of overfitting at this point.";
  }
  if (tl[i] > tl[i - 1] && vl[i] > vl[i - 1]) {
    return "Both train and validation loss rose — possible instability or learning rate too high.";
  }
  return null;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

function formatRunDate(ms: number | null): string {
  if (ms == null) return "—";
  try {
    return new Date(ms).toLocaleString(undefined, {
      dateStyle: "short",
      timeStyle: "short",
    });
  } catch {
    return "—";
  }
}

function useDisplayedRuns(
  runs: RunListItem[],
  exp: ExperimentFilters,
  search: string,
  valMin: string,
  valMax: string,
  testBceMin: string,
  testBceMax: string,
  datePreset: DatePreset,
  sort: RunSort,
  testResults: Record<string, TestEvalResult>,
): RunListItem[] {
  return useMemo(() => {
    const q = search.trim().toLowerCase();
    const minV = valMin === "" ? null : Number(valMin);
    const maxV = valMax === "" ? null : Number(valMax);
    const tMin = testBceMin === "" ? null : Number(testBceMin);
    const tMax = testBceMax === "" ? null : Number(testBceMax);
    const now = Date.now();
    const cutoff =
      datePreset === "24h" ? now - 86400000 : datePreset === "7d" ? now - 7 * 86400000 : null;

    let list = runs.filter((r) => {
      if (!runMatchesExperiment(r, exp)) return false;
      if (q && !r.relative_path.toLowerCase().includes(q)) return false;
      if (minV != null && !Number.isNaN(minV) && r.final_val_loss != null && r.final_val_loss < minV) {
        return false;
      }
      if (maxV != null && !Number.isNaN(maxV) && r.final_val_loss != null && r.final_val_loss > maxV) {
        return false;
      }
      if (cutoff != null && r.history_mtime_ms != null && r.history_mtime_ms < cutoff) {
        return false;
      }
      const tr = testResults[r.id];
      const hasTestFilter =
        (tMin != null && !Number.isNaN(tMin)) || (tMax != null && !Number.isNaN(tMax));
      if (hasTestFilter) {
        if (!tr || !("loss" in tr)) return false;
        if (tMin != null && !Number.isNaN(tMin) && tr.loss < tMin) return false;
        if (tMax != null && !Number.isNaN(tMax) && tr.loss > tMax) return false;
      }
      return true;
    });

    const n = (x: number | null, fallback: number) => (x == null ? fallback : x);

    list = [...list].sort((a, b) => {
      switch (sort) {
        case "val_loss_asc":
          return n(a.final_val_loss, Infinity) - n(b.final_val_loss, Infinity);
        case "val_loss_desc":
          return n(b.final_val_loss, -Infinity) - n(a.final_val_loss, -Infinity);
        case "train_loss_asc":
          return n(a.final_train_loss, Infinity) - n(b.final_train_loss, Infinity);
        case "train_loss_desc":
          return n(b.final_train_loss, -Infinity) - n(a.final_train_loss, -Infinity);
        case "test_bce_asc": {
          const la = testLossForSort(a, testResults);
          const lb = testLossForSort(b, testResults);
          const na = Number.isNaN(la) ? Infinity : la;
          const nb = Number.isNaN(lb) ? Infinity : lb;
          return na - nb;
        }
        case "test_bce_desc": {
          const la = testLossForSort(a, testResults);
          const lb = testLossForSort(b, testResults);
          const na = Number.isNaN(la) ? -Infinity : la;
          const nb = Number.isNaN(lb) ? -Infinity : lb;
          return nb - na;
        }
        case "newest":
          return n(b.history_mtime_ms, 0) - n(a.history_mtime_ms, 0);
        case "path_asc":
          return a.relative_path.localeCompare(b.relative_path);
        default:
          return 0;
      }
    });
    return list;
  }, [
    runs,
    exp,
    search,
    valMin,
    valMax,
    testBceMin,
    testBceMax,
    datePreset,
    sort,
    testResults,
  ]);
}

export default function App() {
  const [tab, setTab] = useState<"data" | "replay" | "live" | "grid">("replay");
  const [runsRoot, setRunsRoot] = useState("temp");
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [detail, setDetail] = useState<RunDetailResponse | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [epochCap, setEpochCap] = useState(1);
  const [compareDetails, setCompareDetails] = useState<RunDetailResponse[]>([]);

  const [runSearch, setRunSearch] = useState("");
  const [valLossMin, setValLossMin] = useState("");
  const [valLossMax, setValLossMax] = useState("");
  const [testBceMin, setTestBceMin] = useState("");
  const [testBceMax, setTestBceMax] = useState("");
  const [datePreset, setDatePreset] = useState<DatePreset>("any");
  const [runSort, setRunSort] = useState<RunSort>("newest");

  const [fTrainPath, setFTrainPath] = useState("");
  const [fLayers, setFLayers] = useState("");
  const [fEpochs, setFEpochs] = useState("");
  const [fLr, setFLr] = useState("");
  const [fSeed, setFSeed] = useState("");
  const [fBatch, setFBatch] = useState("");
  const [fOptimizer, setFOptimizer] = useState("");
  const [fPatience, setFPatience] = useState("");

  const [dataTrainPath, setDataTrainPath] = useState("datasets/data.csv");
  const [dataTestPath, setDataTestPath] = useState("");
  const [gridTestPathsRaw, setGridTestPathsRaw] = useState("");

  const [datasetsScanRoot, setDatasetsScanRoot] = useState("datasets");
  const [datasetCsvFiles, setDatasetCsvFiles] = useState<string[]>([]);
  const [datasetsResolvedRoot, setDatasetsResolvedRoot] = useState<string | null>(null);
  const [datasetsListBusy, setDatasetsListBusy] = useState(false);
  const [datasetsListError, setDatasetsListError] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, TestEvalResult>>({});
  const [replayEvalBusy, setReplayEvalBusy] = useState(false);
  const [replayEvalError, setReplayEvalError] = useState<string | null>(null);

  const expFilters = useMemo<ExperimentFilters>(
    () => ({
      trainPath: fTrainPath,
      layers: fLayers,
      epochs: fEpochs,
      lr: fLr,
      seed: fSeed,
      batch: fBatch,
      optimizer: fOptimizer,
      patience: fPatience,
    }),
    [fTrainPath, fLayers, fEpochs, fLr, fSeed, fBatch, fOptimizer, fPatience],
  );

  const experimentRuns = useMemo(
    () => runs.filter((r) => runMatchesExperiment(r, expFilters)),
    [runs, expFilters],
  );

  const optTrainPaths = useMemo(() => uniqSortedStrings(runs.map((r) => r.config_train_path)), [runs]);
  const optLayers = useMemo(() => uniqSortedStrings(runs.map((r) => r.config_layers_str)), [runs]);
  const optEpochs = useMemo(() => uniqSortedNumberKeys(runs.map((r) => r.config_epochs)), [runs]);
  const optLrs = useMemo(() => uniqSortedNumberKeys(runs.map((r) => r.config_learning_rate)), [runs]);
  const optSeeds = useMemo(() => uniqSortedNumberKeys(runs.map((r) => r.config_seed)), [runs]);
  const optBatches = useMemo(() => uniqSortedNumberKeys(runs.map((r) => r.config_batch_size)), [runs]);
  const optOptim = useMemo(() => uniqSortedStrings(runs.map((r) => r.config_optimizer)), [runs]);
  const optPatience = useMemo(() => uniqSortedNumberKeys(runs.map((r) => r.config_patience)), [runs]);

  const displayedRuns = useDisplayedRuns(
    runs,
    expFilters,
    runSearch,
    valLossMin,
    valLossMax,
    testBceMin,
    testBceMax,
    datePreset,
    runSort,
    testResults,
  );

  useEffect(() => {
    setTestResults({});
  }, [runsRoot]);

  useEffect(() => {
    setTestResults({});
  }, [dataTestPath]);

  const refreshDatasetList = useCallback(async () => {
    setDatasetsListError(null);
    setDatasetsListBusy(true);
    try {
      const d = await fetchJson<DatasetListResponse>(
        `/api/datasets?root=${encodeURIComponent(datasetsScanRoot)}`,
      );
      setDatasetCsvFiles(d.files);
      setDatasetsResolvedRoot(d.datasets_root);
    } catch (e) {
      setDatasetsListError(e instanceof Error ? e.message : String(e));
      setDatasetCsvFiles([]);
      setDatasetsResolvedRoot(null);
    } finally {
      setDatasetsListBusy(false);
    }
  }, [datasetsScanRoot]);

  useEffect(() => {
    void refreshDatasetList();
  }, [refreshDatasetList]);

  const runReplayTestEval = useCallback(async () => {
    const tp = dataTestPath.trim();
    if (!tp || experimentRuns.length === 0) return;
    setReplayEvalBusy(true);
    setReplayEvalError(null);
    try {
      const res = await fetchJson<EvaluateTestResponse>("/api/runs/evaluate-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          root: runsRoot,
          test_path: tp,
          run_paths: experimentRuns.map((r) => r.id),
        }),
      });
      setTestResults((prev) => ({ ...prev, ...res.results }));
    } catch (e) {
      setReplayEvalError(e instanceof Error ? e.message : String(e));
    } finally {
      setReplayEvalBusy(false);
    }
  }, [runsRoot, dataTestPath, experimentRuns]);

  const loadRuns = useCallback(async () => {
    setRunsError(null);
    try {
      const data = await fetchJson<RunListResponse>(
        `/api/runs?root=${encodeURIComponent(runsRoot)}`,
      );
      setRuns(data.runs);
      if (data.runs.length) {
        setSelectedId((id) => {
          if (id && data.runs.some((r) => r.id === id)) return id;
          return data.runs[0].id;
        });
      } else {
        setSelectedId(null);
      }
    } catch (e) {
      setRunsError(e instanceof Error ? e.message : String(e));
    }
  }, [runsRoot]);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailError(null);
    void (async () => {
      try {
        const d = await fetchJson<RunDetailResponse>(
          `/api/runs/${encodeURIComponent(selectedId)}?root=${encodeURIComponent(runsRoot)}`,
        );
        if (!cancelled) {
          setDetail(d);
          setEpochCap(maxEpochs(d.history));
        }
      } catch (e) {
        if (!cancelled) {
          setDetailError(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedId, runsRoot]);

  useEffect(() => {
    if (!compareIds.length) {
      setCompareDetails([]);
      return;
    }
    let cancelled = false;
    void (async () => {
      const out: RunDetailResponse[] = [];
      for (const id of compareIds) {
        try {
          const d = await fetchJson<RunDetailResponse>(
            `/api/runs/${encodeURIComponent(id)}?root=${encodeURIComponent(runsRoot)}`,
          );
          out.push(d);
        } catch {
          /* skip */
        }
      }
      if (!cancelled) setCompareDetails(out);
    })();
    return () => {
      cancelled = true;
    };
  }, [compareIds, runsRoot]);

  const series = useMemo(() => {
    if (!detail) return [];
    return buildSeries(detail.history, epochCap);
  }, [detail, epochCap]);

  const compareValLossData = useMemo(() => {
    if (!compareDetails.length) return [];
    const maxE = Math.max(
      ...compareDetails.map((d) => d.history.history_val_loss.length),
      1,
    );
    const rows: Record<string, number | string>[] = [];
    for (let i = 0; i < maxE; i++) {
      const row: Record<string, number | string> = { epoch: i + 1 };
      compareDetails.forEach((d, j) => {
        const v = d.history.history_val_loss[i];
        if (v !== undefined) {
          row[`val_${j}`] = v;
        }
      });
      rows.push(row);
    }
    return rows;
  }, [compareDetails]);

  const hint = detail ? annotationForEpoch(detail.history, epochCap) : null;

  const [liveParentDir, setLiveParentDir] = useState("");
  const [liveValRatio, setLiveValRatio] = useState(0.2);
  const [liveLayersStr, setLiveLayersStr] = useState("24,24,12");
  const [liveEpochs, setLiveEpochs] = useState(8);
  const [liveLr, setLiveLr] = useState(0.01);
  const [liveSeed, setLiveSeed] = useState(42);
  const [liveBatch, setLiveBatch] = useState(0);
  const [liveOptimizer, setLiveOptimizer] = useState<"sgd" | "rmsprop">("rmsprop");
  const [livePatience, setLivePatience] = useState(0);
  const [liveSample, setLiveSample] = useState(1);
  const [liveLayerPreset, setLiveLayerPreset] = useState("");
  const [liveValRatioPreset, setLiveValRatioPreset] = useState("");
  const [liveBatchPreset, setLiveBatchPreset] = useState("");
  const [liveSeedPreset, setLiveSeedPreset] = useState("");
  const [liveBusy, setLiveBusy] = useState(false);
  const [liveLessonMode, setLiveLessonMode] = useState(false);
  const [lessonPlayback, setLessonPlayback] = useState<{
    manifest: LessonReplayManifest;
    steps: LessonReplayStep[];
  } | null>(null);
  const [liveLog, setLiveLog] = useState<string[]>([]);
  const [liveBatchPoints, setLiveBatchPoints] = useState<
    { step: number; loss: number; epoch: number }[]
  >([]);
  const [liveEpochPoints, setLiveEpochPoints] = useState<
    { epoch: number; train_loss: number; val_loss?: number }[]
  >([]);
  const [liveLastBatch, setLiveLastBatch] = useState<{
    grad_norm_per_layer: number[];
    weight_delta_norm_per_layer: number[];
    epoch: number;
    batch_index: number;
  } | null>(null);
  const [liveTestEval, setLiveTestEval] = useState<{
    loss: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    test_path: string;
    run_dir: string;
  } | null>(null);

  const parseLiveLayers = (s: string): number[] =>
    s
      .split(/[\s,]+/)
      .map((x) => Number(x.trim()))
      .filter((n) => !Number.isNaN(n) && n > 0);

  const startLive = async () => {
    const layers = parseLiveLayers(liveLayersStr);
    if (layers.length < 2) {
      setLiveLog([`layers: need at least 2 positive integers (got "${liveLayersStr}")`]);
      return;
    }
    setLiveBusy(true);
    setLiveLog([]);
    setLiveBatchPoints([]);
    setLiveEpochPoints([]);
    setLiveLastBatch(null);
    setLiveTestEval(null);
    setLessonPlayback(null);
    try {
      const et = dataTestPath.trim();
      const body: Record<string, string | number | boolean | number[] | undefined> = {
        train_path: dataTrainPath,
        parent_dir: liveParentDir.trim() || runsRoot,
        val_ratio: liveValRatio,
        layers,
        epochs: liveEpochs,
        learning_rate: liveLr,
        seed: liveSeed,
        batch_size: liveBatch,
        optimizer: liveOptimizer,
        patience: livePatience,
        telemetry_sample_every_n_batches: liveSample,
      };
      if (et) {
        body.eval_test_path = et;
      }
      if (liveLessonMode) {
        body.lesson_mode = true;
      }
      const { session_id } = await fetchJson<{ session_id: string }>("/api/live/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const wantLessonReplay = liveLessonMode;
      setLiveLog((l) => [
        ...l,
        `session ${session_id}`,
        ...(wantLessonReplay
          ? [
              "Lesson mode on — the fullscreen replay opens right after the \"done\" line (no extra click). If nothing happens, read any red error lines below.",
            ]
          : []),
      ]);
      let step = 0;
      const es = new EventSource(`/api/live/stream/${session_id}`);
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as SseEvent;
          if (msg.type === "batch") {
            step += 1;
            setLiveBatchPoints((p) => [...p, { step, loss: msg.loss, epoch: msg.epoch }]);
            setLiveLastBatch({
              grad_norm_per_layer: msg.grad_norm_per_layer,
              weight_delta_norm_per_layer: msg.weight_delta_norm_per_layer,
              epoch: msg.epoch,
              batch_index: msg.batch_index,
            });
          } else if (msg.type === "epoch") {
            setLiveEpochPoints((p) => [
              ...p,
              {
                epoch: msg.epoch,
                train_loss: msg.train.loss,
                val_loss: msg.val?.loss,
              },
            ]);
          } else if (msg.type === "test_eval") {
            setLiveTestEval({
              loss: msg.loss,
              accuracy: msg.accuracy,
              precision: msg.precision,
              recall: msg.recall,
              f1: msg.f1,
              test_path: msg.test_path,
              run_dir: msg.run_dir,
            });
            setLiveLog((l) => [
              ...l,
              `test BCE ${msg.loss.toFixed(4)} (acc ${msg.accuracy.toFixed(3)}) on ${msg.test_path}`,
            ]);
          } else if (msg.type === "done") {
            setLiveLog((l) => [
              ...l,
              `done: ${msg.epochs_ran} epochs in ${msg.elapsed_seconds.toFixed(2)}s`,
            ]);
            es.close();
            setLiveBusy(false);
            const lm = msg.lesson_manifest;
            const ls = msg.lesson_steps;
            const rd = msg.lesson_replay_run_dir;
            const stepsOk = Array.isArray(ls) && ls.length > 0;
            if (lm != null && stepsOk) {
              setLessonPlayback({ manifest: lm, steps: ls });
              if (wantLessonReplay) {
                setLiveLog((l) => [...l, "Lesson replay: opening fullscreen (data sent with done)."]);
              }
            } else if (rd) {
              setLiveLog((l) => [...l, "Lesson replay: loading from disk (payload was large for SSE)…"]);
              void (async () => {
                try {
                  const data = await fetchJson<{
                    lesson_manifest: LessonReplayManifest;
                    lesson_steps: LessonReplayStep[];
                  }>(`/api/live/lesson-replay?run_dir=${encodeURIComponent(rd)}`);
                  setLessonPlayback({
                    manifest: data.lesson_manifest,
                    steps: data.lesson_steps,
                  });
                  if (wantLessonReplay) {
                    setLiveLog((l) => [...l, "Lesson replay: opening fullscreen (loaded from run_dir)."]);
                  }
                } catch (e) {
                  setLiveLog((l) => [...l, `lesson replay fetch failed: ${String(e)}`]);
                }
              })();
            }
            if (wantLessonReplay) {
              const hasPointer = (lm != null && stepsOk) || (rd != null && rd !== "");
              if (!hasPointer) {
                setLiveLog((l) => [
                  ...l,
                  "Lesson mode was on, but this \"done\" event had no lesson_manifest / lesson_steps / lesson_replay_run_dir. Use a current API build, or confirm the POST body included lesson_mode: true (see Network tab).",
                ]);
              }
            }
          } else if (msg.type === "error") {
            setLiveLog((l) => [...l, `error: ${msg.message}`]);
            es.close();
            setLiveBusy(false);
          }
        } catch (err) {
          const n = ev.data?.length ?? 0;
          setLiveLog((l) => [
            ...l,
            `SSE parse error (${n} chars): ${err instanceof Error ? err.message : String(err)}`,
          ]);
          if (wantLessonReplay) {
            setLiveLog((l) => [
              ...l,
              "Lesson mode: this usually means the \"done\" event was not valid JSON in the browser (e.g. NaN/Infinity in older server responses). Restart the visualizer API and hard-refresh the page.",
            ]);
          }
        }
      };
      es.onerror = () => {
        es.close();
        setLiveBusy(false);
      };
    } catch (e) {
      setLiveLog((l) => [...l, String(e)]);
      setLiveBusy(false);
    }
  };

  const toggleCompare = (id: string) => {
    setCompareIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  };

  const compareSelectAllRef = useRef<HTMLInputElement>(null);
  const compareSelectAllState = useMemo(() => {
    const n = displayedRuns.length;
    if (n === 0) return { all: false, some: false };
    let on = 0;
    for (const r of displayedRuns) {
      if (compareIds.includes(r.id)) on++;
    }
    return {
      all: on === n,
      some: on > 0 && on < n,
    };
  }, [displayedRuns, compareIds]);

  useEffect(() => {
    const el = compareSelectAllRef.current;
    if (el) el.indeterminate = compareSelectAllState.some;
  }, [compareSelectAllState.some]);

  const toggleCompareAll = useCallback(() => {
    setCompareIds((prev) => {
      const displayedSet = new Set(displayedRuns.map((r) => r.id));
      const allOn =
        displayedRuns.length > 0 && displayedRuns.every((r) => prev.includes(r.id));
      if (allOn) {
        return prev.filter((id) => !displayedSet.has(id));
      }
      const inPrev = new Set(prev);
      const toAdd = displayedRuns.map((r) => r.id).filter((id) => !inPrev.has(id));
      return toAdd.length ? [...prev, ...toAdd] : prev;
    });
  }, [displayedRuns]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>MLP training visualizer</h1>
        <p className="hint">
          Start the API with <code>mlp-visualizer</code> (port 8765). For dev UI, run{" "}
          <code>npm install && npm run dev</code> in <code>visualizer/frontend</code> (Vite proxies{" "}
          <code>/api</code>).
        </p>
      </header>

      <div className="tabs">
        <button
          type="button"
          className={tab === "data" ? "active" : ""}
          onClick={() => setTab("data")}
        >
          Data
        </button>
        <button
          type="button"
          className={tab === "replay" ? "active" : ""}
          onClick={() => setTab("replay")}
        >
          Replay runs
        </button>
        <button
          type="button"
          className={tab === "live" ? "active" : ""}
          onClick={() => setTab("live")}
        >
          Live training
        </button>
        <button
          type="button"
          className={tab === "grid" ? "active" : ""}
          onClick={() => setTab("grid")}
        >
          Grid search
        </button>
      </div>

      <div className="toolbar">
        <label>
          Runs root
          <input
            type="text"
            value={runsRoot}
            onChange={(e) => setRunsRoot(e.target.value)}
            style={{ width: "14rem" }}
          />
        </label>
        <button type="button" onClick={() => void loadRuns()}>
          Refresh runs
        </button>
      </div>
      {runsError && <p className="hint error">{runsError}</p>}

      {tab === "data" && (
        <DataManagementPanel
          scanRoot={datasetsScanRoot}
          onScanRootChange={setDatasetsScanRoot}
          relativeFiles={datasetCsvFiles}
          resolvedRootLabel={datasetsResolvedRoot}
          onRefreshList={() => void refreshDatasetList()}
          listBusy={datasetsListBusy}
          listError={datasetsListError}
        />
      )}

      {tab === "replay" && (
        <>
          <div className="panel">
            <h2>
              Runs ({displayedRuns.length} shown · {experimentRuns.length} match experiment filters ·{" "}
              {runs.length} total)
            </h2>
            <p className="hint">
              Use experiment dropdowns to narrow configs, then run test-set evaluation. The table applies
              all filters including test BCE after scores are computed.
            </p>
            <div className="row" style={{ marginBottom: "0.75rem", flexWrap: "wrap", gap: "0.5rem" }}>
              <label>
                Train CSV (config)
                <select value={fTrainPath} onChange={(e) => setFTrainPath(e.target.value)}>
                  <option value="">Any</option>
                  {optTrainPaths.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Layers
                <select value={fLayers} onChange={(e) => setFLayers(e.target.value)}>
                  <option value="">Any</option>
                  {optLayers.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Epochs (cfg)
                <select value={fEpochs} onChange={(e) => setFEpochs(e.target.value)}>
                  <option value="">Any</option>
                  {optEpochs.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                LR
                <select value={fLr} onChange={(e) => setFLr(e.target.value)}>
                  <option value="">Any</option>
                  {optLrs.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <select value={fSeed} onChange={(e) => setFSeed(e.target.value)}>
                  <option value="">Any</option>
                  {optSeeds.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Batch
                <select value={fBatch} onChange={(e) => setFBatch(e.target.value)}>
                  <option value="">Any</option>
                  {optBatches.map((p) => (
                    <option key={p} value={p}>
                      {p === "0" ? "full (0)" : p}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Optimizer
                <select value={fOptimizer} onChange={(e) => setFOptimizer(e.target.value)}>
                  <option value="">Any</option>
                  {optOptim.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Patience
                <select value={fPatience} onChange={(e) => setFPatience(e.target.value)}>
                  <option value="">Any</option>
                  {optPatience.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="row" style={{ marginBottom: "0.75rem", flexWrap: "wrap", alignItems: "flex-end" }}>
              <DatasetCsvPickRow
                label="Test CSV (binary CE)"
                pathValue={dataTestPath}
                onPathChange={setDataTestPath}
                scanRoot={datasetsScanRoot}
                relativeFiles={datasetCsvFiles}
                inputPlaceholder="datasets/…/test.csv"
              />
              <button
                type="button"
                disabled={replayEvalBusy || !dataTestPath.trim() || experimentRuns.length === 0}
                onClick={() => void runReplayTestEval()}
              >
                {replayEvalBusy ? "Evaluating…" : `Evaluate ${experimentRuns.length} runs`}
              </button>
            </div>
            {replayEvalError && <p className="hint error">{replayEvalError}</p>}
            <div className="row" style={{ marginBottom: "0.75rem", flexWrap: "wrap" }}>
              <label>
                Search path
                <input
                  type="text"
                  placeholder="substring…"
                  value={runSearch}
                  onChange={(e) => setRunSearch(e.target.value)}
                  style={{ width: "12rem" }}
                />
              </label>
              <label>
                Sort
                <select
                  value={runSort}
                  onChange={(e) => setRunSort(e.target.value as RunSort)}
                >
                  <option value="newest">Newest first</option>
                  <option value="val_loss_asc">Val loss (low → high)</option>
                  <option value="val_loss_desc">Val loss (high → low)</option>
                  <option value="train_loss_asc">Train loss (low → high)</option>
                  <option value="train_loss_desc">Train loss (high → low)</option>
                  <option value="test_bce_asc">Test BCE (low → high)</option>
                  <option value="test_bce_desc">Test BCE (high → low)</option>
                  <option value="path_asc">Path A–Z</option>
                </select>
              </label>
              <label>
                Updated
                <select
                  value={datePreset}
                  onChange={(e) => setDatePreset(e.target.value as DatePreset)}
                >
                  <option value="any">Any time</option>
                  <option value="24h">Last 24h</option>
                  <option value="7d">Last 7 days</option>
                </select>
              </label>
              <label>
                Min val loss
                <input
                  type="number"
                  step={0.01}
                  placeholder="—"
                  value={valLossMin}
                  onChange={(e) => setValLossMin(e.target.value)}
                  style={{ width: "5.5rem" }}
                />
              </label>
              <label>
                Max val loss
                <input
                  type="number"
                  step={0.01}
                  placeholder="—"
                  value={valLossMax}
                  onChange={(e) => setValLossMax(e.target.value)}
                  style={{ width: "5.5rem" }}
                />
              </label>
              <label>
                Min test BCE
                <input
                  type="number"
                  step={0.001}
                  placeholder="—"
                  value={testBceMin}
                  onChange={(e) => setTestBceMin(e.target.value)}
                  style={{ width: "5.5rem" }}
                />
              </label>
              <label>
                Max test BCE
                <input
                  type="number"
                  step={0.001}
                  placeholder="—"
                  value={testBceMax}
                  onChange={(e) => setTestBceMax(e.target.value)}
                  style={{ width: "5.5rem" }}
                />
              </label>
            </div>
            <div className="run-table-wrap">
              <table className="run-table">
                <thead>
                  <tr>
                    <th style={{ width: 40 }} title="Compare">
                      <input
                        ref={compareSelectAllRef}
                        type="checkbox"
                        checked={compareSelectAllState.all}
                        onChange={toggleCompareAll}
                        disabled={displayedRuns.length === 0}
                        title="Select all for compare"
                        aria-label="Select all runs for compare"
                      />
                    </th>
                    <th style={{ width: 36 }} aria-label="Select" />
                    <th>Path</th>
                    <th>Val loss</th>
                    <th>Train loss</th>
                    <th>Epochs</th>
                    <th>Test BCE</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {displayedRuns.map((r) => {
                    const testDisp = formatTestCell(r, testResults);
                    return (
                    <tr
                      key={r.id}
                      className={selectedId === r.id ? "selected" : ""}
                      onClick={() => setSelectedId(r.id)}
                    >
                      <td onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={compareIds.includes(r.id)}
                          onChange={() => toggleCompare(r.id)}
                          title="Compare"
                          aria-label="Compare"
                        />
                      </td>
                      <td onClick={(e) => e.stopPropagation()}>
                        <input
                          type="radio"
                          name="run-select"
                          checked={selectedId === r.id}
                          onChange={() => setSelectedId(r.id)}
                          aria-label={`Select ${r.relative_path}`}
                        />
                      </td>
                      <td className="path-cell" title={r.relative_path}>
                        {r.relative_path}
                      </td>
                      <td className="num">
                        {r.final_val_loss != null ? r.final_val_loss.toFixed(4) : "—"}
                      </td>
                      <td className="num">
                        {r.final_train_loss != null ? r.final_train_loss.toFixed(4) : "—"}
                      </td>
                      <td className="num">{r.epochs_ran ?? "—"}</td>
                      <td className="num" title={testDisp.title}>
                        {testDisp.cell}
                      </td>
                      <td className="num" style={{ fontSize: "0.78rem" }}>
                        {formatRunDate(r.history_mtime_ms)}
                      </td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {detailError && <p className="hint error">{detailError}</p>}

          {detail && (
            <div className="panel">
              <h2>Run detail</h2>
              <p className="hint" style={{ wordBreak: "break-all" }}>
                {detail.run_path}
              </p>
              {detail.run_config && (
                <div className="metric-grid" style={{ marginTop: "0.5rem" }}>
                  <div className="metric-card">layers: {detail.run_config.layers.join(" → ")}</div>
                  <div className="metric-card">lr: {detail.run_config.learning_rate}</div>
                  <div className="metric-card">epochs (config): {detail.run_config.epochs}</div>
                  <div className="metric-card">optimizer: {detail.run_config.optimizer}</div>
                  <div className="metric-card">batch: {detail.run_config.batch_size || "full"}</div>
                  <div className="metric-card">patience: {detail.run_config.patience}</div>
                </div>
              )}

              <div className="slider-row">
                <label>Epoch (replay)</label>
                <input
                  type="range"
                  min={1}
                  max={maxEpochs(detail.history)}
                  value={epochCap}
                  onChange={(e) => setEpochCap(Number(e.target.value))}
                />
                <span className="hint">
                  {epochCap} / {maxEpochs(detail.history)}
                </span>
              </div>
              {hint && <div className="annotation">{hint}</div>}

              <h2>Loss</h2>
              <div className="chart-box">
                <ResponsiveContainer>
                  <LineChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis dataKey="epoch" stroke={CHART_AXIS} />
                    <YAxis stroke={CHART_AXIS} />
                    <Tooltip contentStyle={CHART_TOOLTIP} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="train_loss"
                      name="train loss"
                      stroke="#5b90f5"
                      dot={false}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      name="val loss"
                      stroke="#e87872"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <h2>Accuracy</h2>
              <div className="chart-box">
                <ResponsiveContainer>
                  <LineChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis dataKey="epoch" stroke={CHART_AXIS} />
                    <YAxis domain={[0, 1]} stroke={CHART_AXIS} />
                    <Tooltip contentStyle={CHART_TOOLTIP} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="train_acc"
                      name="train acc"
                      stroke="#5cb87a"
                      dot={false}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="val_acc"
                      name="val acc"
                      stroke="#e4b04a"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {compareDetails.length > 1 && (
            <div className="panel">
              <h2>Compare validation loss</h2>
              <div className="chart-box" style={{ height: 300 }}>
                <ResponsiveContainer>
                  <LineChart data={compareValLossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis dataKey="epoch" stroke={CHART_AXIS} />
                    <YAxis stroke={CHART_AXIS} />
                    <Tooltip contentStyle={CHART_TOOLTIP} />
                    <Legend />
                    {compareDetails.map((d, j) => (
                      <Line
                        key={d.run_path}
                        type="monotone"
                        dataKey={`val_${j}`}
                        name={compareIds[j] ?? d.run_path}
                        stroke={COLORS[j % COLORS.length]}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}

      {tab === "live" && (
        <div className="panel">
          <h2>Live training (SSE)</h2>
          <p className="hint">
            All fields match <code>POST /api/live/train</code>. Parent directory defaults to the toolbar
            &quot;Runs root&quot; when left empty. If you set a test CSV, the run is saved and test BCE is
            streamed after training.
          </p>
          <div className="row" style={{ flexDirection: "column", alignItems: "stretch", gap: "0.65rem" }}>
            <DatasetCsvPickRow
              label="Train CSV path"
              pathValue={dataTrainPath}
              onPathChange={setDataTrainPath}
              scanRoot={datasetsScanRoot}
              relativeFiles={datasetCsvFiles}
            />
            <label>
              Parent dir (saved runs / live output) — empty = use toolbar Runs root (
              <code>{runsRoot}</code>){" "}
              <input
                type="text"
                value={liveParentDir}
                onChange={(e) => setLiveParentDir(e.target.value)}
                placeholder={runsRoot}
                style={{ width: "100%", maxWidth: "32rem" }}
              />
            </label>
            <div className="row" style={{ flexWrap: "wrap", alignItems: "flex-end" }}>
              <label>
                Val ratio (train split)
                <input
                  type="number"
                  step={0.05}
                  min={0.05}
                  max={0.49}
                  value={liveValRatio}
                  onChange={(e) => setLiveValRatio(Number(e.target.value))}
                  style={{ width: "5.5rem" }}
                />
              </label>
              <label>
                Preset
                <select
                  value={liveValRatioPreset}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v) setLiveValRatio(Number(v));
                    setLiveValRatioPreset("");
                  }}
                >
                  <option value="">—</option>
                  {LIVE_VAL_RATIO_PRESETS.map((p) => (
                    <option key={p.value} value={String(p.value)}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="row" style={{ flexWrap: "wrap", alignItems: "flex-end" }}>
              <label>
                Hidden layers (comma-separated widths, min 2)
                <input
                  type="text"
                  value={liveLayersStr}
                  onChange={(e) => setLiveLayersStr(e.target.value)}
                  placeholder="24,24,12"
                  style={{ width: "12rem" }}
                />
              </label>
              <label>
                Architecture preset
                <select
                  value={liveLayerPreset}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v) setLiveLayersStr(v);
                    setLiveLayerPreset("");
                  }}
                >
                  <option value="">—</option>
                  {LIVE_LAYER_PRESETS.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <DatasetCsvPickRow
              label="Optional test CSV (binary CE after train; persists model)"
              pathValue={dataTestPath}
              onPathChange={setDataTestPath}
              scanRoot={datasetsScanRoot}
              relativeFiles={datasetCsvFiles}
              inputPlaceholder="Leave empty to skip test eval"
            />
            <div className="row" style={{ flexWrap: "wrap" }}>
              <label>
                Epochs{" "}
                <input
                  type="number"
                  min={1}
                  value={liveEpochs}
                  onChange={(e) => setLiveEpochs(Number(e.target.value))}
                />
              </label>
              <label>
                Learning rate{" "}
                <input
                  type="number"
                  step={0.001}
                  min={0.0001}
                  value={liveLr}
                  onChange={(e) => setLiveLr(Number(e.target.value))}
                />
              </label>
              <label>
                Seed{" "}
                <input
                  type="number"
                  value={liveSeed}
                  onChange={(e) => setLiveSeed(Number(e.target.value))}
                  style={{ width: "5rem" }}
                />
              </label>
              <label>
                Seed preset
                <select
                  value={liveSeedPreset}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v !== "") setLiveSeed(Number(v));
                    setLiveSeedPreset("");
                  }}
                >
                  <option value="">—</option>
                  {LIVE_SEED_PRESETS.map((p) => (
                    <option key={p.value} value={String(p.value)}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="row" style={{ flexWrap: "wrap", alignItems: "flex-end" }}>
              <label>
                Batch size (0 = full batch){" "}
                <input
                  type="number"
                  min={0}
                  value={liveBatch}
                  onChange={(e) => setLiveBatch(Number(e.target.value))}
                  style={{ width: "5rem" }}
                />
              </label>
              <label>
                Batch preset
                <select
                  value={liveBatchPreset}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v !== "") setLiveBatch(Number(v));
                    setLiveBatchPreset("");
                  }}
                >
                  <option value="">—</option>
                  {LIVE_BATCH_PRESETS.map((p) => (
                    <option key={p.label} value={String(p.value)}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Optimizer{" "}
                <select
                  value={liveOptimizer}
                  onChange={(e) => setLiveOptimizer(e.target.value as "sgd" | "rmsprop")}
                >
                  <option value="rmsprop">rmsprop</option>
                  <option value="sgd">sgd</option>
                </select>
              </label>
              <label>
                Patience (early stopping, 0 = off){" "}
                <input
                  type="number"
                  min={0}
                  value={livePatience}
                  onChange={(e) => setLivePatience(Number(e.target.value))}
                  style={{ width: "4.5rem" }}
                />
              </label>
              <label>
                Batch telemetry every N{" "}
                <input
                  type="number"
                  min={1}
                  value={liveSample}
                  onChange={(e) => setLiveSample(Number(e.target.value))}
                  style={{ width: "4rem" }}
                />
              </label>
            </div>
            <label className="hint" style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
              <input
                type="checkbox"
                checked={liveLessonMode}
                onChange={(e) => setLiveLessonMode(e.target.checked)}
                disabled={liveBusy}
                style={{ marginTop: "0.2rem" }}
              />
              <span>
                <strong>Lesson mode</strong> — check this before &quot;Start live training&quot;. When training
                finishes, a fullscreen replay opens automatically (no extra button). Micro-steps are capped on
                the server so huge runs do not freeze the browser.
              </span>
            </label>
            <button type="button" disabled={liveBusy} onClick={() => void startLive()}>
              {liveBusy ? "Training…" : "Start live training"}
            </button>
            <ul className="hint">
              {liveLog.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>

          {liveTestEval && (
            <div className="metric-grid" style={{ marginTop: "0.75rem" }}>
              <div className="metric-card" style={{ gridColumn: "1 / -1" }}>
                Test set ({liveTestEval.test_path}) — BCE {liveTestEval.loss.toFixed(4)} · acc{" "}
                {liveTestEval.accuracy.toFixed(4)} · prec {liveTestEval.precision.toFixed(4)} · rec{" "}
                {liveTestEval.recall.toFixed(4)} · F1 {liveTestEval.f1.toFixed(4)}
              </div>
              <div className="metric-card" style={{ gridColumn: "1 / -1", fontSize: "0.8rem" }}>
                Saved run: {liveTestEval.run_dir}
              </div>
            </div>
          )}

          {liveLastBatch && (
            <div className="metric-grid" style={{ marginTop: "0.75rem" }}>
              <div className="metric-card" style={{ gridColumn: "1 / -1" }}>
                Last batch (epoch {liveLastBatch.epoch}, batch {liveLastBatch.batch_index}): grad L2 per
                layer —{" "}
                {liveLastBatch.grad_norm_per_layer
                  .map((g, i) => `L${i}:${g.toExponential(2)}`)
                  .join(" · ")}
              </div>
              <div className="metric-card" style={{ gridColumn: "1 / -1" }}>
                Weight update L2 per layer —{" "}
                {liveLastBatch.weight_delta_norm_per_layer
                  .map((w, i) => `L${i}:${w.toExponential(2)}`)
                  .join(" · ")}
              </div>
            </div>
          )}

          {liveBatchPoints.length > 0 && (
            <>
              <h3>Batch loss (streamed)</h3>
              <div className="chart-box chart-box-sm">
                <ResponsiveContainer>
                  <LineChart data={liveBatchPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis dataKey="step" stroke={CHART_AXIS} />
                    <YAxis stroke={CHART_AXIS} />
                    <Tooltip contentStyle={CHART_TOOLTIP} />
                    <Line
                      type="monotone"
                      dataKey="loss"
                      name="batch CE"
                      stroke="#b279a2"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {liveEpochPoints.length > 0 && (
            <>
              <h3>Epoch train / val loss (streamed)</h3>
              <div className="chart-box" style={{ height: 260 }}>
                <ResponsiveContainer>
                  <LineChart data={liveEpochPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                    <XAxis dataKey="epoch" stroke={CHART_AXIS} />
                    <YAxis stroke={CHART_AXIS} />
                    <Tooltip contentStyle={CHART_TOOLTIP} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="train_loss"
                      name="train"
                      stroke="#5b90f5"
                      dot={false}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      name="val"
                      stroke="#e87872"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      )}

      {tab === "grid" && (
        <GridSearchPanel
          runsRoot={runsRoot}
          onRunsRootChange={setRunsRoot}
          onRefreshRuns={loadRuns}
          trainPath={dataTrainPath}
          onTrainPathChange={setDataTrainPath}
          testPathsRaw={gridTestPathsRaw}
          onTestPathsRawChange={setGridTestPathsRaw}
          datasetsScanRoot={datasetsScanRoot}
          datasetCsvFiles={datasetCsvFiles}
        />
      )}

      {lessonPlayback && (
        <FullscreenLessonPlayback
          manifest={lessonPlayback.manifest}
          steps={lessonPlayback.steps}
          onClose={() => setLessonPlayback(null)}
        />
      )}
    </div>
  );
}
