import { useState } from "react";
import type { BestSummaryPayload, GridSseEvent } from "../types";
import { DatasetCsvPickRow } from "./DatasetCsvPickRow";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

function basename(p: string): string {
  const parts = p.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || p;
}

function joinDatasetsScanPath(scanRoot: string, rel: string): string {
  const r = scanRoot.replace(/\/+$/, "");
  const p = rel.replace(/^\/+/, "");
  return p ? `${r}/${p}` : r;
}

function pickLoss(row: Record<string, unknown>): number | undefined {
  const v = row.val_loss;
  const t = row.test_loss;
  if (typeof v === "number") return v;
  if (typeof t === "number") return t;
  return undefined;
}

function BestCards({ board }: { board: BestSummaryPayload }) {
  const br = board.best_by_recall as Record<string, unknown> | undefined;
  const bf = board.best_by_f1 as Record<string, unknown> | undefined;
  const bl = board.best_by_val_loss as Record<string, unknown> | undefined;
  return (
    <div className="leaderboard-grid">
      <div className="leader-card">
        <strong>Best val / test loss</strong>
        {bl ? (
          <>
            <div className="value">
              {typeof bl.val_loss === "number"
                ? bl.val_loss.toFixed(4)
                : typeof bl.test_loss === "number"
                  ? bl.test_loss.toFixed(4)
                  : "—"}
            </div>
            <div className="hint" style={{ marginTop: "0.35rem", wordBreak: "break-all" }}>
              {basename(String(bl.run_dir ?? ""))}
            </div>
          </>
        ) : (
          <span className="hint">—</span>
        )}
      </div>
      <div className="leader-card">
        <strong>Best F1</strong>
        {bf ? (
          <>
            <div className="value">
              {typeof bf.val_f1 === "number"
                ? bf.val_f1.toFixed(4)
                : typeof bf.test_f1 === "number"
                  ? bf.test_f1.toFixed(4)
                  : "—"}
            </div>
            <div className="hint" style={{ marginTop: "0.35rem", wordBreak: "break-all" }}>
              {basename(String(bf.run_dir ?? ""))}
            </div>
          </>
        ) : (
          <span className="hint">—</span>
        )}
      </div>
      <div className="leader-card">
        <strong>Best recall</strong>
        {br ? (
          <>
            <div className="value">
              {typeof br.val_recall === "number"
                ? br.val_recall.toFixed(4)
                : typeof br.test_recall === "number"
                  ? br.test_recall.toFixed(4)
                  : "—"}
            </div>
            <div className="hint" style={{ marginTop: "0.35rem", wordBreak: "break-all" }}>
              {basename(String(br.run_dir ?? ""))}
            </div>
          </>
        ) : (
          <span className="hint">—</span>
        )}
      </div>
    </div>
  );
}

function TopLossTable({ board }: { board: BestSummaryPayload }) {
  const rows = board.top_5_by_val_loss as Record<string, unknown>[] | undefined;
  if (!rows?.length) return null;
  return (
    <div style={{ marginTop: "1rem" }}>
      <h3>Top runs by loss (so far)</h3>
      <div className="run-table-wrap" style={{ maxHeight: 200 }}>
        <table className="run-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Loss</th>
              <th>Run</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="num">{i + 1}</td>
                <td className="num">{pickLoss(r)?.toFixed(4) ?? "—"}</td>
                <td className="path-cell" title={String(r.run_dir ?? "")}>
                  {basename(String(r.run_dir ?? ""))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type Props = {
  runsRoot: string;
  onRunsRootChange: (v: string) => void;
  onRefreshRuns: () => void;
  trainPath: string;
  onTrainPathChange: (v: string) => void;
  testPathsRaw: string;
  onTestPathsRawChange: (v: string) => void;
  datasetsScanRoot: string;
  datasetCsvFiles: string[];
};

export function GridSearchPanel({
  runsRoot,
  onRunsRootChange,
  onRefreshRuns,
  trainPath,
  onTrainPathChange,
  testPathsRaw,
  onTestPathsRawChange,
  datasetsScanRoot,
  datasetCsvFiles,
}: Props) {
  const [epochs, setEpochs] = useState(8);
  const [valRatio, setValRatio] = useState(0.2);
  const [gridMode, setGridMode] = useState<"quick" | "full">("quick");
  const [testPathPicker, setTestPathPicker] = useState("");
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [progress, setProgress] = useState<{ index: number; total: number } | null>(null);
  const [leaderboard, setLeaderboard] = useState<BestSummaryPayload | null>(null);
  const [finalSummary, setFinalSummary] = useState<BestSummaryPayload | null>(null);
  const [gridParent, setGridParent] = useState<string | null>(null);

  const start = async () => {
    setBusy(true);
    setLog([]);
    setProgress(null);
    setLeaderboard(null);
    setFinalSummary(null);
    setGridParent(null);
    const test_paths = testPathsRaw
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    try {
      const body = {
        train_path: trainPath,
        epochs,
        val_ratio: valRatio,
        seed: 42,
        parent_dir: runsRoot,
        test_paths,
        grid_mode: gridMode,
      };
      const { session_id } = await fetchJson<{ session_id: string }>("/api/live/best", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setLog((l) => [...l, `session ${session_id}`]);
      const es = new EventSource(`/api/live/best/stream/${session_id}`);
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as GridSseEvent;
          if (msg.type === "grid_started") {
            setGridParent(msg.parent_dir);
            setLog((l) => [
              ...l,
              `Grid ${msg.grid_mode}: ${msg.total} combos (full CLI grid = ${msg.full_grid_combos})`,
            ]);
            setProgress({ index: 0, total: msg.total });
          } else if (msg.type === "grid_combo_done") {
            setProgress({ index: msg.index, total: msg.total });
            setLeaderboard(msg.leaderboard);
            setLog((l) => [...l, `[${msg.index}/${msg.total}] ${msg.run_name}`]);
          } else if (msg.type === "grid_final") {
            setFinalSummary(msg.summary);
            setGridParent(msg.parent_dir);
            setLog((l) => [...l, `Finished — best run: ${basename(msg.best_run_dir)}`]);
          } else if (msg.type === "done") {
            es.close();
            setBusy(false);
          } else if (msg.type === "error") {
            setLog((l) => [...l, `error: ${msg.message}`]);
            es.close();
            setBusy(false);
          }
        } catch {
          /* ignore */
        }
      };
      es.onerror = () => {
        es.close();
        setBusy(false);
      };
    } catch (e) {
      setLog((l) => [...l, String(e)]);
      setBusy(false);
    }
  };

  const pct =
    progress && progress.total > 0
      ? Math.round((100 * progress.index) / progress.total)
      : 0;

  return (
    <div className="panel">
      <h2>Hyperparameter grid (best search)</h2>
      <p className="hint">
        Runs the same style of search as <code>mlp-train --best</code>. Quick mode uses a small grid
        (~12 runs). Full mode runs the entire CLI grid (hundreds of trainings); use only when you
        intend to wait a long time.
      </p>

      <div className="row" style={{ flexDirection: "column", alignItems: "stretch", gap: "0.75rem" }}>
        <DatasetCsvPickRow
          label="Train CSV path"
          pathValue={trainPath}
          onPathChange={onTrainPathChange}
          scanRoot={datasetsScanRoot}
          relativeFiles={datasetCsvFiles}
        />
        <label>
          Optional test CSV paths (comma or newline separated)
          <textarea
            value={testPathsRaw}
            onChange={(e) => onTestPathsRawChange(e.target.value)}
            rows={2}
            style={{
              width: "100%",
              maxWidth: "28rem",
              background: "var(--bg)",
              border: "1px solid var(--border)",
              color: "var(--text)",
              borderRadius: "var(--radius-sm)",
              padding: "0.45rem 0.55rem",
              fontFamily: "inherit",
            }}
          />
        </label>
        <label>
          Add one test path from <code>{datasetsScanRoot}</code>
          <select
            value={testPathPicker}
            onChange={(e) => {
              const rel = e.target.value;
              if (!rel) return;
              const full = joinDatasetsScanPath(datasetsScanRoot, rel);
              onTestPathsRawChange(
                testPathsRaw.trim() ? `${testPathsRaw.trim()}\n${full}` : full,
              );
              setTestPathPicker("");
            }}
          >
            <option value="">—</option>
            {datasetCsvFiles.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
        <div className="row">
          <label>
            Epochs
            <input
              type="number"
              min={1}
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
          </label>
          <label>
            Val ratio
            <input
              type="number"
              step={0.05}
              min={0.05}
              max={0.5}
              value={valRatio}
              onChange={(e) => setValRatio(Number(e.target.value))}
            />
          </label>
          <label>
            Grid
            <select
              value={gridMode}
              onChange={(e) => setGridMode(e.target.value as "quick" | "full")}
            >
              <option value="quick">Quick (~12)</option>
              <option value="full">Full (~600)</option>
            </select>
          </label>
        </div>
        {gridMode === "full" && (
          <div className="annotation">
            Full grid runs hundreds of sequential trainings. Confirm you want this before starting.
          </div>
        )}
        <button type="button" disabled={busy} onClick={() => void start()}>
          {busy ? "Grid running…" : "Start grid search"}
        </button>
      </div>

      {progress && (
        <div style={{ marginTop: "1rem" }}>
          <div className="hint">
            Progress: {progress.index} / {progress.total}
          </div>
          <div className="progress-bar">
            <span style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      {leaderboard && Object.keys(leaderboard).length > 0 && (
        <>
          <h3>Best so far</h3>
          <BestCards board={leaderboard} />
          <TopLossTable board={leaderboard} />
        </>
      )}

      {finalSummary && Object.keys(finalSummary).length > 0 && (
        <>
          <h3>Final summary (after test eval if configured)</h3>
          <BestCards board={finalSummary} />
          <TopLossTable board={finalSummary} />
        </>
      )}

      {gridParent && (
        <div className="row" style={{ marginTop: "1rem" }}>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              onRunsRootChange(gridParent);
              onRefreshRuns();
            }}
          >
            Use grid folder as runs root
          </button>
        </div>
      )}

      <ul className="hint" style={{ marginTop: "1rem" }}>
        {log.map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ul>
    </div>
  );
}
