import type { LucideIcon } from "lucide-react";
import { ChevronDown } from "lucide-react";
import type { ReactNode } from "react";

type DisclosureProps = {
  title: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
};

export function Disclosure({ title, children, defaultOpen, className }: DisclosureProps) {
  return (
    <details
      className={`disclosure${className ? ` ${className}` : ""}`}
      {...(defaultOpen !== undefined ? { defaultOpen } : {})}
    >
      <summary>
        {title}
        <ChevronDown className="disclosure-chevron" size={16} aria-hidden />
      </summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}

type FormGridProps = {
  children: ReactNode;
  wide?: boolean;
  className?: string;
};

export function FormGrid({ children, wide, className }: FormGridProps) {
  const g = wide ? "form-grid form-grid--wide" : "form-grid";
  return <div className={className ? `${g} ${className}` : g}>{children}</div>;
}

type FieldLabelProps = {
  children: ReactNode;
  htmlFor?: string;
  icon?: LucideIcon;
  className?: string;
};

export function FieldLabel({ children, htmlFor, icon: Icon, className }: FieldLabelProps) {
  const row = (
    <span className="field-label-row">
      {Icon ? <Icon size={14} strokeWidth={2} aria-hidden /> : null}
      <span>{children}</span>
    </span>
  );
  return (
    <label className={className ? `field-label ${className}` : "field-label"} htmlFor={htmlFor}>
      {row}
    </label>
  );
}

type CalloutVariant = "insight" | "warning" | "info";

type CalloutProps = {
  variant: CalloutVariant;
  title: ReactNode;
  children: ReactNode;
  icon?: LucideIcon;
  className?: string;
};

export function Callout({ variant, title, children, icon: Icon, className }: CalloutProps) {
  const mod = `callout callout--${variant}`;
  return (
    <div className={className ? `${mod} ${className}` : mod} role="note">
      <div className="callout-title">
        {Icon ? <Icon size={16} strokeWidth={2} aria-hidden /> : null}
        {title}
      </div>
      <div className="callout-body">{children}</div>
    </div>
  );
}

type LogPanelProps = {
  lines: string[];
  title?: string;
  emptyLabel?: string;
  className?: string;
};

export function LogPanel({ lines, title = "Event log", emptyLabel = "No events yet.", className }: LogPanelProps) {
  const empty = lines.length === 0;
  const base = `log-panel${empty ? " log-panel--empty" : ""}`;
  return (
    <div className={className ? `${base} ${className}` : base}>
      <div className="log-panel-header">{title}</div>
      {empty ? (
        <p className="log-panel-body">{emptyLabel}</p>
      ) : (
        <ul className="log-panel-body">
          {lines.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function FormStack({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={className ? `form-stack ${className}` : "form-stack"}>{children}</div>;
}
