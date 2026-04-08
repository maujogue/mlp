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
};

/** Dropdown of CSVs under `scanRoot` plus a text field (paths relative to project CWD). */
export function DatasetCsvPickRow({
  label,
  pathValue,
  onPathChange,
  scanRoot,
  relativeFiles,
  inputPlaceholder,
}: Props) {
  const relForSelect = (() => {
    const prefix = `${scanRoot.replace(/\/+$/, "")}/`;
    const v = pathValue.trim();
    if (!v.startsWith(prefix)) return "";
    return v.slice(prefix.length);
  })();

  return (
    <div className="row" style={{ flexWrap: "wrap", alignItems: "flex-end", gap: "0.5rem" }}>
      <label>
        {label}
        <input
          type="text"
          value={pathValue}
          onChange={(e) => onPathChange(e.target.value)}
          placeholder={inputPlaceholder ?? `${scanRoot}/…`}
          style={{ width: "min(100%, 28rem)" }}
        />
      </label>
      <label>
        Pick under <code>{scanRoot}</code>
        <select
          value={relativeFiles.includes(relForSelect) ? relForSelect : ""}
          onChange={(e) => {
            const rel = e.target.value;
            if (rel) onPathChange(joinScanRoot(scanRoot, rel));
          }}
        >
          <option value="">—</option>
          {relativeFiles.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
