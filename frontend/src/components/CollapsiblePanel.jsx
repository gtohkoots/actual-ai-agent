import { useState } from "react";

const joinClassNames = (...values) => values.filter(Boolean).join(" ");

export default function CollapsiblePanel({
  sectionLabel,
  title,
  className,
  defaultOpen = false,
  summary = null,
  collapsedLabel = "Show details",
  expandedLabel = "Hide details",
  collapsible = true,
  children,
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <article className={joinClassNames("panel collapsible-panel", className)}>
      <div className="panel-header collapsible-panel__header">
        <div>
          {sectionLabel ? <p className="section-label">{sectionLabel}</p> : null}
          <h3>{title}</h3>
        </div>
      </div>

      {summary ? (
        <div className="collapsible-panel__summary">
          <div className="collapsible-panel__summary-copy">{summary}</div>
          {collapsible ? (
            <button
              className="ghost-button collapsible-panel__toggle"
              type="button"
              onClick={() => setIsOpen((current) => !current)}
            >
              {isOpen ? expandedLabel : collapsedLabel}
            </button>
          ) : null}
        </div>
      ) : collapsible ? (
        <div className="collapsible-panel__summary collapsible-panel__summary--toggle-only">
          <button
            className="ghost-button collapsible-panel__toggle"
            type="button"
            onClick={() => setIsOpen((current) => !current)}
          >
            {isOpen ? expandedLabel : collapsedLabel}
          </button>
        </div>
      ) : null}

      {!collapsible || isOpen ? <div className="collapsible-panel__body">{children}</div> : null}
    </article>
  );
}
