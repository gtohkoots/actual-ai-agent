import { useEffect, useMemo, useState } from "react";

import { Bot, MessageCircle, X } from "lucide-react";

import ChatPanel from "./ChatPanel";

const MIN_WIDTH = 360;
const MIN_HEIGHT = 420;
const DEFAULT_WIDTH = 460;
const DEFAULT_HEIGHT = 760;
const VIEWPORT_MARGIN = 24;
const SHELL_BOTTOM_OFFSET = 72;
const NOOP = () => {};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function getMaxWidth() {
  if (typeof window === "undefined") return DEFAULT_WIDTH;
  return Math.max(MIN_WIDTH, window.innerWidth - VIEWPORT_MARGIN);
}

function getMaxHeight() {
  if (typeof window === "undefined") return DEFAULT_HEIGHT;
  return Math.max(MIN_HEIGHT, window.innerHeight - SHELL_BOTTOM_OFFSET - 16);
}

function getInitialSize() {
  return {
    width: clamp(DEFAULT_WIDTH, MIN_WIDTH, getMaxWidth()),
    height: clamp(DEFAULT_HEIGHT, MIN_HEIGHT, getMaxHeight()),
  };
}

function AssistantShell({
  open,
  mode,
  onModeChange,
  onClose,
  card,
  analysisWindow,
  seedMessage = "",
  seedMessageId = "",
  onSeedConsumed = () => {},
}) {
  const [shellSize, setShellSize] = useState(() => getInitialSize());
  const [resizeState, setResizeState] = useState(null);

  useEffect(() => {
    function handleViewportResize() {
      setShellSize((current) => ({
        width: clamp(current.width, MIN_WIDTH, getMaxWidth()),
        height: clamp(current.height, MIN_HEIGHT, getMaxHeight()),
      }));
    }

    window.addEventListener("resize", handleViewportResize);
    return () => window.removeEventListener("resize", handleViewportResize);
  }, []);

  useEffect(() => {
    if (!resizeState) {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      return undefined;
    }

    const cursorMap = {
      left: "ew-resize",
      top: "ns-resize",
      corner: "nwse-resize",
    };
    document.body.style.userSelect = "none";
    document.body.style.cursor = cursorMap[resizeState.handle] || "";

    function handlePointerMove(event) {
      const deltaX = event.clientX - resizeState.startX;
      const deltaY = event.clientY - resizeState.startY;
      const nextWidth = resizeState.handle === "top"
        ? resizeState.startWidth
        : clamp(resizeState.startWidth - deltaX, MIN_WIDTH, getMaxWidth());
      const nextHeight = resizeState.handle === "left"
        ? resizeState.startHeight
        : clamp(resizeState.startHeight - deltaY, MIN_HEIGHT, getMaxHeight());

      setShellSize({
        width: nextWidth,
        height: nextHeight,
      });
    }

    function handlePointerUp() {
      setResizeState(null);
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [resizeState]);

  const shellStyle = useMemo(
    () => ({
      width: `${shellSize.width}px`,
      height: `${shellSize.height}px`,
    }),
    [shellSize.height, shellSize.width]
  );

  function startResize(handle, event) {
    event.preventDefault();
    setResizeState({
      handle,
      startX: event.clientX,
      startY: event.clientY,
      startWidth: shellSize.width,
      startHeight: shellSize.height,
    });
  }

  if (!open) return null;

  const plannerSelected = mode === "planner";

  return (
    <section className="assistant-shell" aria-label="Assistant chat" style={shellStyle}>
      <div
        className="assistant-shell__resize-handle assistant-shell__resize-handle--left"
        onPointerDown={(event) => startResize("left", event)}
        aria-hidden="true"
      />
      <div
        className="assistant-shell__resize-handle assistant-shell__resize-handle--top"
        onPointerDown={(event) => startResize("top", event)}
        aria-hidden="true"
      />
      <div
        className="assistant-shell__resize-handle assistant-shell__resize-handle--corner"
        onPointerDown={(event) => startResize("corner", event)}
        aria-hidden="true"
      />
      <div className="assistant-shell__header">
        <div>
          <p className="section-label">Assistant</p>
          <h3>{plannerSelected ? "Planner chat" : "Analysis chat"}</h3>
        </div>
        <button
          className="ghost-button assistant-shell__close"
          type="button"
          onClick={onClose}
          aria-label="Close assistant"
          title="Close assistant"
        >
          <X size={16} aria-hidden="true" />
        </button>
      </div>

      <div className="assistant-shell__mode-toggle" role="tablist" aria-label="Assistant mode">
        <button
          className={`assistant-shell__mode-pill ${!plannerSelected ? "active" : ""}`}
          type="button"
          role="tab"
          aria-selected={!plannerSelected}
          onClick={() => onModeChange("analysis")}
        >
          <MessageCircle size={14} aria-hidden="true" />
          <span>Analysis</span>
        </button>
        <button
          className={`assistant-shell__mode-pill ${plannerSelected ? "active" : ""}`}
          type="button"
          role="tab"
          aria-selected={plannerSelected}
          onClick={() => onModeChange("planner")}
        >
          <Bot size={14} aria-hidden="true" />
          <span>Planner</span>
        </button>
      </div>

      <div className="assistant-shell__body">
        <ChatPanel
          key={`assistant-shell-${mode}`}
          card={card}
          analysisWindow={analysisWindow}
          mode={plannerSelected ? "planner" : "legacy"}
          layout="shell"
          seedMessage={plannerSelected ? "" : seedMessage}
          seedMessageId={plannerSelected ? "" : seedMessageId}
          onSeedConsumed={plannerSelected ? NOOP : onSeedConsumed}
        />
      </div>
    </section>
  );
}

export default AssistantShell;
