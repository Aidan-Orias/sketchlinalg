import type { BenchmarkOptions } from "../types/benchmark";

type BenchmarkControlsProps = {
  options: BenchmarkOptions;
  sketchDimension: number;
  alpha: number;
  repeats: number;
  seed: number;
  running: boolean;
  onSketchDimensionChange: (value: number) => void;
  onAlphaChange: (value: number) => void;
  onRepeatsChange: (value: number) => void;
  onSeedChange: (value: number) => void;
  onRun: () => void;
};

export function BenchmarkControls({
  options,
  sketchDimension,
  alpha,
  repeats,
  seed,
  running,
  onSketchDimensionChange,
  onAlphaChange,
  onRepeatsChange,
  onSeedChange,
  onRun
}: BenchmarkControlsProps) {
  const alphaPreset = options.alpha_values.includes(alpha) ? String(alpha) : "custom";

  return (
    <section className="tool-section">
      <div className="section-heading">
        <h2>Benchmark Controls</h2>
        <span>Ridge + CountSketch</span>
      </div>

      <div className="control-grid">
        <label className="field-group field-wide">
          <span>Sketch dimension</span>
          <div className="range-input-pair">
            <input
              type="range"
              min={options.sketch_dimension_min}
              max={options.sketch_dimension_max}
              step={options.sketch_dimension_step}
              value={sketchDimension}
              disabled={running}
              onChange={(event) =>
                onSketchDimensionChange(Number(event.target.value))
              }
            />
            <input
              className="numeric-input"
              type="number"
              min={1}
              max={250000}
              step={options.sketch_dimension_step}
              value={sketchDimension}
              disabled={running}
              onChange={(event) =>
                onSketchDimensionChange(Number(event.target.value))
              }
            />
          </div>
        </label>

        <label className="field-group field-alpha">
          <span>Ridge alpha</span>
          <div className="inline-inputs">
            <select
              value={alphaPreset}
              disabled={running}
              onChange={(event) => {
                if (event.target.value !== "custom") {
                  onAlphaChange(Number(event.target.value));
                }
              }}
            >
              {options.alpha_values.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
              <option value="custom">Custom</option>
            </select>
            <input
              className="numeric-input"
              type="number"
              min={0}
              step={0.01}
              value={alpha}
              disabled={running}
              onChange={(event) => onAlphaChange(Number(event.target.value))}
            />
          </div>
        </label>

        <label className="field-group field-small">
          <span>Repeats</span>
          <input
            className="numeric-input"
            type="number"
            min={1}
            max={25}
            step={1}
            value={repeats}
            disabled={running}
            onChange={(event) => onRepeatsChange(Number(event.target.value))}
          />
        </label>

        <label className="field-group field-seed">
          <span>Random seed</span>
          <input
            className="numeric-input"
            type="number"
            step={1}
            value={seed}
            disabled={running}
            onChange={(event) => onSeedChange(Number(event.target.value))}
          />
        </label>
      </div>

      <button className="run-button" onClick={onRun} disabled={running}>
        {running ? "Running" : "Run Benchmark"}
      </button>
    </section>
  );
}
