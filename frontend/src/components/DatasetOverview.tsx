import type { DatasetSummary } from "../types/benchmark";
import { formatCompact, formatNumber, formatPercent } from "../utils/format";

type DatasetOverviewProps = {
  summary: DatasetSummary | null;
  error: string | null;
};

export function DatasetOverview({ summary, error }: DatasetOverviewProps) {
  if (error) {
    return (
      <section className="content-section">
        <h2>Dataset</h2>
        <p className="muted">{error}</p>
      </section>
    );
  }

  if (!summary) {
    return (
      <section className="content-section">
        <h2>Dataset</h2>
        <p className="muted">Loading dataset summary.</p>
      </section>
    );
  }

  return (
    <section className="content-section">
      <div className="section-heading">
        <h2>{summary.name}</h2>
        <span>{summary.task}</span>
      </div>
      <div className="dataset-grid">
        <div className="stat-card">
          <span>Train Rows</span>
          <strong>{formatNumber(summary.train_rows)}</strong>
        </div>
        <div className="stat-card">
          <span>Test Rows</span>
          <strong>{formatNumber(summary.test_rows)}</strong>
        </div>
        <div className="stat-card">
          <span>Features</span>
          <strong>{formatNumber(summary.features)}</strong>
        </div>
        <div className="stat-card">
          <span>Train NNZ</span>
          <strong>{formatCompact(summary.train_nnz)}</strong>
        </div>
      </div>
      <dl className="dataset-details">
        <div>
          <dt>Train density</dt>
          <dd>{formatPercent(summary.train_density)}</dd>
        </div>
        <div>
          <dt>Test density</dt>
          <dd>{formatPercent(summary.test_density)}</dd>
        </div>
        <div>
          <dt>Target mean</dt>
          <dd>{summary.target_mean.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Target spread</dt>
          <dd>{summary.target_std.toFixed(3)}</dd>
        </div>
      </dl>
    </section>
  );
}
