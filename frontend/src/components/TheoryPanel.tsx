import type { DatasetSummary } from "../types/benchmark";
import { DatasetOverview } from "./DatasetOverview";
import { MathBlock } from "./MathBlock";

type TheoryPanelProps = {
  summary: DatasetSummary | null;
  summaryError: string | null;
};

export function TheoryPanel({ summary, summaryError }: TheoryPanelProps) {
  return (
    <aside className="theory-panel">
      <div className="brand-line">sketchlinalg</div>
      <section className="content-section hero-copy">
        <h1>CountSketch Ridge Benchmarks</h1>
        <p>
          Matrix sketching replaces a tall training matrix{" "}
          <MathBlock>X</MathBlock> with a smaller matrix{" "}
          <MathBlock>SX</MathBlock>, preserving enough geometry to fit a useful
          model while reducing the cost of the regression solve.
        </p>
      </section>

      <section className="content-section">
        <h2>Matrix Sketching</h2>
        <p>
          For <MathBlock>{"X \\in \\mathbb{R}^{n \\times p}"}</MathBlock>, a
          sketch matrix{" "}
          <MathBlock>{"S \\in \\mathbb{R}^{d \\times n}"}</MathBlock> compresses
          rows so the model fits on <MathBlock>SX</MathBlock> and{" "}
          <MathBlock>Sy</MathBlock>. When <MathBlock>{"d \\ll n"}</MathBlock>,
          fitting can be faster, but the approximation can change the fitted
          coefficients and prediction error.
        </p>
        <div className="equation-strip">
          <MathBlock block>
            {"\\min_w \\lVert Xw - y \\rVert_2^2 + \\alpha \\lVert w \\rVert_2^2"}
          </MathBlock>
          <MathBlock block>
            {"SX \\in \\mathbb{R}^{d \\times p}, \\qquad d \\ll n"}
          </MathBlock>
        </div>
      </section>

      <section className="content-section">
        <h2>CountSketch</h2>
        <p>
          CountSketch hashes each original row into one of{" "}
          <MathBlock>d</MathBlock> buckets and flips it by a random sign. The
          projection is sparse, so the benchmark can compress the E2006 sparse
          design matrix without constructing a dense random projection.
        </p>
        <div className="sketch-diagram" aria-label="CountSketch matrix diagram">
          <svg viewBox="0 0 620 170" role="img">
            <rect x="20" y="28" width="94" height="116" rx="8" />
            <text x="67" y="92">S</text>
            <rect x="150" y="16" width="128" height="140" rx="8" />
            <text x="214" y="92">X</text>
            <path d="M122 86 L144 86" />
            <path d="M286 86 L320 86" />
            <rect x="328" y="44" width="124" height="84" rx="8" />
            <text x="390" y="92">SX</text>
            <path d="M460 86 L494 86" />
            <rect x="502" y="52" width="98" height="68" rx="8" />
            <text x="551" y="92">fit</text>
          </svg>
        </div>
      </section>

      <DatasetOverview summary={summary} error={summaryError} />
    </aside>
  );
}
