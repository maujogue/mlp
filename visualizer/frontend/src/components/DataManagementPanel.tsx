import { useState } from "react";
import type { PrepareDatasetResponse, SplitDatasetResponse } from "../types";
import { DatasetCsvPickRow } from "./DatasetCsvPickRow";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

type Props = {
  scanRoot: string;
  onScanRootChange: (v: string) => void;
  relativeFiles: string[];
  resolvedRootLabel: string | null;
  onRefreshList: () => void;
  listBusy: boolean;
  listError: string | null;
};

export function DataManagementPanel({
  scanRoot,
  onScanRootChange,
  relativeFiles,
  resolvedRootLabel,
  onRefreshList,
  listBusy,
  listError,
}: Props) {
  const [prepareSource, setPrepareSource] = useState("");
  const [prepareOutput, setPrepareOutput] = useState("");
  const [prepareBusy, setPrepareBusy] = useState(false);
  const [prepareMsg, setPrepareMsg] = useState<string | null>(null);

  const [splitPrepared, setSplitPrepared] = useState("");
  const [splitTestSize, setSplitTestSize] = useState(0.2);
  const [splitBusy, setSplitBusy] = useState(false);
  const [splitMsg, setSplitMsg] = useState<string | null>(null);

  const runPrepare = async () => {
    const src = prepareSource.trim();
    if (!src) {
      setPrepareMsg("Set a source CSV path.");
      return;
    }
    setPrepareBusy(true);
    setPrepareMsg(null);
    try {
      const body: Record<string, string> = { source: src };
      const out = prepareOutput.trim();
      if (out) body.output = out;
      const res = await fetchJson<PrepareDatasetResponse>("/api/datasets/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setPrepareMsg(`Wrote prepared file: ${res.output}`);
      onRefreshList();
    } catch (e) {
      setPrepareMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setPrepareBusy(false);
    }
  };

  const runSplit = async () => {
    const p = splitPrepared.trim();
    if (!p) {
      setSplitMsg("Set a prepared CSV path.");
      return;
    }
    setSplitBusy(true);
    setSplitMsg(null);
    try {
      const res = await fetchJson<SplitDatasetResponse>("/api/datasets/split", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prepared_path: p, test_size: splitTestSize }),
      });
      setSplitMsg(`Train: ${res.train_path}\nTest: ${res.test_path}`);
      onRefreshList();
    } catch (e) {
      setSplitMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setSplitBusy(false);
    }
  };

  return (
    <div className="panel">
      <h2>Data preparation</h2>
      <p className="hint">
        List CSVs under a folder (default <code>datasets/</code>), prepare a raw file (same as{" "}
        <code>mlp-prepare-dataset</code>), then split a prepared file into{" "}
        <code>datasets/&lt;stem&gt;/train.csv</code> and <code>test.csv</code> (same as{" "}
        <code>mlp-split</code>). Use the pickers on other tabs to point training and evaluation at these
        paths.
      </p>

      <div className="row" style={{ flexWrap: "wrap", alignItems: "flex-end", gap: "0.5rem" }}>
        <label>
          Scan folder for CSV list
          <input
            type="text"
            value={scanRoot}
            onChange={(e) => onScanRootChange(e.target.value)}
            style={{ width: "10rem" }}
          />
        </label>
        <button type="button" disabled={listBusy} onClick={() => onRefreshList()}>
          {listBusy ? "Loading…" : "Refresh list"}
        </button>
        {resolvedRootLabel && (
          <span className="hint" style={{ wordBreak: "break-all" }}>
            Resolved: {resolvedRootLabel}
          </span>
        )}
      </div>
      {listError && <p className="hint error">{listError}</p>}

      <h3 style={{ marginTop: "1.25rem" }}>Prepare raw → prepared</h3>
      <DatasetCsvPickRow
        label="Source CSV"
        pathValue={prepareSource}
        onPathChange={setPrepareSource}
        scanRoot={scanRoot}
        relativeFiles={relativeFiles}
        inputPlaceholder="datasets/raw.csv"
      />
      <label style={{ display: "block", marginTop: "0.5rem" }}>
        Optional output path (empty = next to source, *_prepared.csv)
        <input
          type="text"
          value={prepareOutput}
          onChange={(e) => setPrepareOutput(e.target.value)}
          style={{ width: "min(100%, 28rem)", marginTop: "0.25rem" }}
        />
      </label>
      <button
        type="button"
        style={{ marginTop: "0.65rem" }}
        disabled={prepareBusy}
        onClick={() => void runPrepare()}
      >
        {prepareBusy ? "Preparing…" : "Prepare dataset"}
      </button>
      {prepareMsg && (
        <pre className="hint" style={{ marginTop: "0.5rem", whiteSpace: "pre-wrap" }}>
          {prepareMsg}
        </pre>
      )}

      <h3 style={{ marginTop: "1.25rem" }}>Split prepared → train / test</h3>
      <DatasetCsvPickRow
        label="Prepared CSV"
        pathValue={splitPrepared}
        onPathChange={setSplitPrepared}
        scanRoot={scanRoot}
        relativeFiles={relativeFiles}
        inputPlaceholder="datasets/data_prepared.csv"
      />
      <div className="row" style={{ marginTop: "0.5rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <label>
          Test fraction
          <input
            type="number"
            step={0.05}
            min={0.05}
            max={0.49}
            value={splitTestSize}
            onChange={(e) => setSplitTestSize(Number(e.target.value))}
            style={{ width: "5.5rem" }}
          />
        </label>
        <button type="button" disabled={splitBusy} onClick={() => void runSplit()}>
          {splitBusy ? "Splitting…" : "Split"}
        </button>
      </div>
      {splitMsg && (
        <pre className="hint" style={{ marginTop: "0.5rem", whiteSpace: "pre-wrap" }}>
          {splitMsg}
        </pre>
      )}

      <h3 style={{ marginTop: "1.25rem" }}>CSV files under scan folder</h3>
      {relativeFiles.length === 0 ? (
        <p className="hint">No CSV files found. Adjust the scan folder or add data.</p>
      ) : (
        <ul className="hint" style={{ maxHeight: 200, overflow: "auto", margin: 0 }}>
          {relativeFiles.map((f) => (
            <li key={f}>
              <code>{f}</code>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
