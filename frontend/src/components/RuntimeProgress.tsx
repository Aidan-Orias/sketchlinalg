import type { CSSProperties } from "react";
import type { BenchmarkPhase } from "../types/benchmark";
import { formatSeconds } from "../utils/format";

export type TrackedPhase =
  | "fit_regular"
  | "construct_sketch"
  | "fit_sketched";

export type PhaseStatus = "pending" | "running" | "complete" | "error";

export type PhaseState = {
  progress: number;
  status: PhaseStatus;
  elapsed_seconds?: number | null;
  message?: string | null;
};

type RuntimeProgressProps = {
  phases: Record<TrackedPhase, PhaseState>;
  statusMessage: string;
  running: boolean;
};

const PHASE_ROWS: Array<{
  phase: TrackedPhase;
  label: string;
  color: string;
}> = [
  { phase: "fit_regular", label: "Regular fit", color: "#2563eb" },
  { phase: "construct_sketch", label: "Sketch construction", color: "#f97316" },
  { phase: "fit_sketched", label: "Sketched fit", color: "#159947" }
];

function phaseLabel(phase: BenchmarkPhase): string {
  return phase.replace(/_/g, " ");
}

export function RuntimeProgress({
  phases,
  statusMessage,
  running
}: RuntimeProgressProps) {
  return (
    <section className="tool-section progress-section">
      <div className="section-heading">
        <h2>Runtime Progress</h2>
        <span>{running ? statusMessage || "Working" : "Idle"}</span>
      </div>

      <div className="progress-stack">
        {PHASE_ROWS.map(({ phase, label, color }) => {
          const state = phases[phase];
          const percent = Math.round(state.progress * 100);
          return (
            <div className="progress-row" key={phase}>
              <div className="progress-meta">
                <span>{label}</span>
                <strong>{percent}%</strong>
              </div>
              <div className="progress-track" aria-label={phaseLabel(phase)}>
                <div
                  className={`progress-fill ${state.status}`}
                  style={
                    {
                      width: `${percent}%`,
                      "--phase-color": color
                    } as CSSProperties
                  }
                />
              </div>
              <div className="progress-foot">
                <span>{state.message ?? state.status}</span>
                <span>
                  {state.elapsed_seconds
                    ? `Elapsed ${formatSeconds(state.elapsed_seconds)}`
                    : ""}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
