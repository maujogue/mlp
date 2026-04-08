import { useState } from "react";
import { FolderOpen } from "lucide-react";
import type { PrepareDatasetResponse, SplitDatasetResponse } from "../types";
import { DatasetCsvPickRow } from "./DatasetCsvPickRow";
import { Disclosure, FieldLabel, FormStack } from "./ui-shell";

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
      <Disclosure title="About data prep">
        <p className="hint" style={{ marginTop: "var(--space-2)" }}>
          List CSVs under a folder (default <code>datasets/</code>), prepare a raw file (same as{" "}
          <code>mlp-prepare-dataset</code>), then split a prepared file into{" "}
          <code>datasets/&lt;stem&gt;/train.csv</code> and <code>test.csv</code> (same as{" "}
          <code>mlp-split</code>). Use the pickers on other tabs to point training and evaluation at these
          paths.
        </p>
      </Disclosure>

      <div className="row" style={{ marginBottom: "var(--space-3)", alignItems: "flex-end", flexWrap: "wrap" }}>
        <div>
          <FieldLabel htmlFor="data-scan-root" icon={FolderOpen}>
            Scan folder for CSV list
          </FieldLabel>
          <input
            id="data-scan-root"
            type="text"
            value={scanRoot}
            onChange={(e) => onScanRootChange(e.target.value)}
            style={{ width: "14rem", marginTop: "var(--space-1)" }}
          />
        </div>
        <button type="button" disabled={listBusy} onClick={() => onRefreshList()}>
          {listBusy ? "Loading…" : "Refresh list"}
        </button>
      </div>
      {resolvedRootLabel && (
        <p className="hint" style={{ wordBreak: "break-all", marginTop: 0 }}>
          Resolved: {resolvedRootLabel}
        </p>
      )}
      {listError && <p className="hint error">{listError}</p>}

      <Disclosure title="Prepare raw → prepared" defaultOpen>
        <FormStack>
          <DatasetCsvPickRow
            idPrefix="data-prepare-src"
            label="Source CSV"
            pathValue={prepareSource}
            onPathChange={setPrepareSource}
            scanRoot={scanRoot}
            relativeFiles={relativeFiles}
            inputPlaceholder="datasets/raw.csv"
          />
          <div>
            <FieldLabel htmlFor="prepare-output-path">
              Optional output path (empty = next to source, *_prepared.csv)
            </FieldLabel>
            <input
              id="prepare-output-path"
              type="text"
              value={prepareOutput}
              onChange={(e) => setPrepareOutput(e.target.value)}
              style={{ width: "min(100%, 28rem)", marginTop: "var(--space-1)" }}
            />
          </div>
          <button type="button" disabled={prepareBusy} onClick={() => void runPrepare()}>
            {prepareBusy ? "Preparing…" : "Prepare dataset"}
          </button>
          {prepareMsg && (
            <pre className="hint" style={{ margin: 0, whiteSpace: "pre-wrap" }}>
              {prepareMsg}
            </pre>
          )}
        </FormStack>
      </Disclosure>

      <Disclosure title="Split prepared → train / test" defaultOpen>
        <FormStack>
          <DatasetCsvPickRow
            idPrefix="data-split-prepared"
            label="Prepared CSV"
            pathValue={splitPrepared}
            onPathChange={setSplitPrepared}
            scanRoot={scanRoot}
            relativeFiles={relativeFiles}
            inputPlaceholder="datasets/data_prepared.csv"
          />
          <div className="row" style={{ alignItems: "flex-end", flexWrap: "wrap", gap: "var(--space-3)" }}>
            <label className="field-label">
              <span className="field-label-row">Test fraction</span>
              <input
                type="number"
                step={0.05}
                min={0.05}
                max={0.49}
                value={splitTestSize}
                onChange={(e) => setSplitTestSize(Number(e.target.value))}
              />
            </label>
            <button type="button" disabled={splitBusy} onClick={() => void runSplit()}>
              {splitBusy ? "Splitting…" : "Split"}
            </button>
          </div>
          {splitMsg && (
            <pre className="hint" style={{ margin: 0, whiteSpace: "pre-wrap" }}>
              {splitMsg}
            </pre>
          )}
        </FormStack>
      </Disclosure>

      <Disclosure title="CSV files under scan folder" defaultOpen>
        {relativeFiles.length === 0 ? (
          <p className="hint" style={{ marginTop: "var(--space-2)" }}>
            No CSV files found. Adjust the scan folder or add data.
          </p>
        ) : (
          <ul className="hint" style={{ maxHeight: 200, overflow: "auto", margin: "var(--space-2) 0 0", paddingLeft: "1.25rem" }}>
            {relativeFiles.map((f) => (
              <li key={f}>
                <code>{f}</code>
              </li>
            ))}
          </ul>
        )}
      </Disclosure>
    </div>
  );
}
