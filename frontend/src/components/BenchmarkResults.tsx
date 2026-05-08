import type { BenchmarkResult } from "../types/benchmark";
import { formatRatio, formatSeconds } from "../utils/format";

type BenchmarkResultsProps = {
  result: BenchmarkResult | null;
};

export function BenchmarkResults({ result }: BenchmarkResultsProps) {
  if (!result) {
    return (
      <section className="tool-section results-section empty-state">
        <div className="section-heading">
          <h2>Benchmarks</h2>
          <span>Awaiting run</span>
        </div>
      </section>
    );
  }

  const maxTime = Math.max(
    result.timings.total_regular,
    result.timings.total_sketched,
    0.001
  );
  const regularWidth = (result.timings.total_regular / maxTime) * 100;
  const sketchConstructionWidth =
    (result.timings.sketch_construction / maxTime) * 100;
  const sketchedFitWidth = (result.timings.sketched_fit / maxTime) * 100;

  return (
    <section className="tool-section results-section">
      <div className="section-heading">
        <h2>Benchmarks</h2>
        <span>d = {result.sketch_dimension}, alpha = {result.alpha}</span>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <span>Total Speedup</span>
          <strong>{formatRatio(result.metrics.total_speedup)}</strong>
        </div>
        <div className="metric-card">
          <span>Fit Speedup</span>
          <strong>{formatRatio(result.metrics.fit_speedup)}</strong>
        </div>
        <div className="metric-card">
          <span>RMSE Ratio</span>
          <strong>{result.metrics.rmse_ratio.toFixed(4)}</strong>
        </div>
        <div className="metric-card">
          <span>Repeats</span>
          <strong>{result.repeats}</strong>
        </div>
      </div>

      <div className="runtime-chart">
        <div className="chart-row">
          <div className="chart-label">Regular</div>
          <div className="chart-track">
            <div
              className="chart-segment regular"
              style={{ width: `${regularWidth}%` }}
            />
          </div>
          <strong>{formatSeconds(result.timings.total_regular)}</strong>
        </div>

        <div className="chart-row">
          <div className="chart-label">Sketched</div>
          <div className="chart-track split">
            <div
              className="chart-segment sketch"
              style={{ width: `${sketchConstructionWidth}%` }}
            />
            <div
              className="chart-segment sketched-fit"
              style={{ width: `${sketchedFitWidth}%` }}
            />
          </div>
          <strong>{formatSeconds(result.timings.total_sketched)}</strong>
        </div>
      </div>

      <div className="legend-row">
        <span>
          <i className="legend-dot regular" /> Regular fit
        </span>
        <span>
          <i className="legend-dot sketch" /> Sketch construction
        </span>
        <span>
          <i className="legend-dot sketched-fit" /> Sketched fit
        </span>
      </div>

      <table className="result-table">
        <tbody>
          <tr>
            <th>Regular fit</th>
            <td>{formatSeconds(result.timings.regular_fit)}</td>
            <td>RMSE {result.metrics.regular_rmse.toFixed(4)}</td>
          </tr>
          <tr>
            <th>Sketch construction</th>
            <td>{formatSeconds(result.timings.sketch_construction)}</td>
            <td>CountSketch d = {result.sketch_dimension}</td>
          </tr>
          <tr>
            <th>Sketched fit</th>
            <td>{formatSeconds(result.timings.sketched_fit)}</td>
            <td>RMSE {result.metrics.sketched_rmse.toFixed(4)}</td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}
