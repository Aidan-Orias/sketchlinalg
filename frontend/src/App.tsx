import { useEffect, useRef, useState } from "react";
import {
  connectBenchmarkProgress,
  fetchBenchmarkOptions,
  fetchDatasetSummary,
  startBenchmark
} from "./api/benchmarkClient";
import { BenchmarkControls } from "./components/BenchmarkControls";
import { BenchmarkResults } from "./components/BenchmarkResults";
import {
  RuntimeProgress
} from "./components/RuntimeProgress";
import type { PhaseState, TrackedPhase } from "./components/RuntimeProgress";
import { TheoryPanel } from "./components/TheoryPanel";
import type {
  BenchmarkOptions,
  BenchmarkProgressEvent,
  BenchmarkResult,
  DatasetSummary
} from "./types/benchmark";

const DEFAULT_OPTIONS: BenchmarkOptions = {
  sketch_dimension_min: 1000,
  sketch_dimension_max: 50000,
  sketch_dimension_step: 500,
  default_sketch_dimension: 5000,
  alpha_values: [1e-6, 0.01, 0.1, 1.0, 5.0, 10.0],
  default_alpha: 1.0,
  default_repeats: 3,
  default_seed: 127
};

const TRACKED_PHASES: TrackedPhase[] = [
  "fit_regular",
  "construct_sketch",
  "fit_sketched"
];

function createInitialPhases(): Record<TrackedPhase, PhaseState> {
  return {
    fit_regular: { progress: 0, status: "pending" },
    construct_sketch: { progress: 0, status: "pending" },
    fit_sketched: { progress: 0, status: "pending" }
  };
}

function isTrackedPhase(phase: string): phase is TrackedPhase {
  return TRACKED_PHASES.includes(phase as TrackedPhase);
}

function phaseStatus(event: BenchmarkProgressEvent): PhaseState["status"] {
  if (event.type === "error") {
    return "error";
  }
  if (event.progress >= 1) {
    return "complete";
  }
  return "running";
}

export default function App() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [options, setOptions] = useState<BenchmarkOptions>(DEFAULT_OPTIONS);
  const [sketchDimension, setSketchDimension] = useState(
    DEFAULT_OPTIONS.default_sketch_dimension
  );
  const [alpha, setAlpha] = useState(DEFAULT_OPTIONS.default_alpha);
  const [repeats, setRepeats] = useState(DEFAULT_OPTIONS.default_repeats);
  const [seed, setSeed] = useState(DEFAULT_OPTIONS.default_seed);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [phases, setPhases] = useState(createInitialPhases);
  const [statusMessage, setStatusMessage] = useState("");
  const [runError, setRunError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let active = true;

    fetchDatasetSummary()
      .then((nextSummary) => {
        if (active) {
          setSummary(nextSummary);
        }
      })
      .catch((error: Error) => {
        if (active) {
          setSummaryError(error.message);
        }
      });

    fetchBenchmarkOptions()
      .then((nextOptions) => {
        if (active) {
          setOptions(nextOptions);
          setSketchDimension(nextOptions.default_sketch_dimension);
          setAlpha(nextOptions.default_alpha);
          setRepeats(nextOptions.default_repeats);
          setSeed(nextOptions.default_seed);
        }
      })
      .catch(() => {
        if (active) {
          setOptions(DEFAULT_OPTIONS);
        }
      });

    return () => {
      active = false;
      socketRef.current?.close();
    };
  }, []);

  function handleProgressEvent(event: BenchmarkProgressEvent) {
    if (event.message) {
      setStatusMessage(event.message);
    }

    if (isTrackedPhase(event.phase)) {
      setPhases((current) => ({
        ...current,
        [event.phase]: {
          progress: event.progress,
          status: phaseStatus(event),
          elapsed_seconds: event.elapsed_seconds,
          message: event.message
        }
      }));
    }

    if (event.type === "complete") {
      if (event.result) {
        setResult(event.result);
      }
      setRunning(false);
      socketRef.current?.close();
    }

    if (event.type === "error") {
      setRunError(event.error ?? event.message ?? "Benchmark failed");
      setRunning(false);
      socketRef.current?.close();
    }
  }

  async function handleRun() {
    socketRef.current?.close();
    setRunning(true);
    setRunError(null);
    setResult(null);
    setStatusMessage("Benchmark queued");
    setPhases(createInitialPhases());

    try {
      const job = await startBenchmark({
        sketch_dimension: Math.max(1, Math.round(sketchDimension)),
        alpha: Math.max(0, alpha),
        repeats: Math.max(1, Math.round(repeats)),
        seed: Number.isFinite(seed) ? Math.round(seed) : null
      });

      socketRef.current = connectBenchmarkProgress(
        job.job_id,
        handleProgressEvent,
        () => {
          setRunError("Lost progress connection");
          setRunning(false);
        }
      );
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Benchmark failed");
      setRunning(false);
    }
  }

  return (
    <main className="app-shell">
      <TheoryPanel summary={summary} summaryError={summaryError} />

      <section className="workbench-panel">
        <BenchmarkControls
          options={options}
          sketchDimension={sketchDimension}
          alpha={alpha}
          repeats={repeats}
          seed={seed}
          running={running}
          onSketchDimensionChange={setSketchDimension}
          onAlphaChange={setAlpha}
          onRepeatsChange={setRepeats}
          onSeedChange={setSeed}
          onRun={handleRun}
        />

        {runError ? <div className="error-banner">{runError}</div> : null}

        <RuntimeProgress
          phases={phases}
          statusMessage={statusMessage}
          running={running}
        />

        <BenchmarkResults result={result} />
      </section>
    </main>
  );
}
