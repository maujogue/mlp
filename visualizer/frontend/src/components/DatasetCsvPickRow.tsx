import { FileText } from "lucide-react";
import { useId } from "react";
import { Disclosure, FieldLabel } from "./ui-shell";

function joinScanRoot(scanRoot: string, rel: string): string {
  const r = scanRoot.replace(/\/+$/, "");
  const p = rel.replace(/^\/+/, "");
  return p ? `${r}/${p}` : r;
}

type Props = {
  label: string;
  pathValue: string;
  onPathChange: (v: string) => void;
  scanRoot: string;
  relativeFiles: string[];
  inputPlaceholder?: string;
  /** Stable prefix for input id when multiple pickers share a screen */
  idPrefix?: string;
};

/** Dropdown of CSVs under `scanRoot` plus a text field (paths relative to project CWD). */
export function DatasetCsvPickRow({
  label,
  pathValue,
  onPathChange,
  scanRoot,
  relativeFiles,
  inputPlaceholder,
  idPrefix = "csv-path",
}: Props) {
  const reactId = useId();
  const pathInputId = `${idPrefix}-${reactId.replace(/:/g, "")}`;

  const relForSelect = (() => {
    const prefix = `${scanRoot.replace(/\/+$/, "")}/`;
    const v = pathValue.trim();
    if (!v.startsWith(prefix)) return "";
    return v.slice(prefix.length);
  })();

  return (
    <div className="form-stack">
      <div>
        <FieldLabel htmlFor={pathInputId} icon={FileText}>
          {label}
        </FieldLabel>
        <input
          id={pathInputId}
          type="text"
          value={pathValue}
          onChange={(e) => onPathChange(e.target.value)}
          placeholder={inputPlaceholder ?? `${scanRoot}/…`}
          style={{ width: "min(100%, 28rem)", marginTop: "var(--space-1)" }}
        />
      </div>
      <Disclosure title={`Pick from scan folder (${scanRoot})`}>
        <label className="field-label" style={{ marginTop: "var(--space-2)" }}>
          <span className="field-label-row">CSV file</span>
          <select
            value={relativeFiles.includes(relForSelect) ? relForSelect : ""}
            onChange={(e) => {
              const rel = e.target.value;
              if (rel) onPathChange(joinScanRoot(scanRoot, rel));
            }}
            style={{ maxWidth: "28rem", marginTop: "var(--space-1)" }}
          >
            <option value="">—</option>
            {relativeFiles.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
      </Disclosure>
    </div>
  );
}
