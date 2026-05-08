export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatCompact(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2
  }).format(value);
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(3)}%`;
}

export function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) {
    return "0.000 s";
  }
  if (value < 0.001) {
    return `${(value * 1000).toFixed(2)} ms`;
  }
  if (value < 1) {
    return `${(value * 1000).toFixed(1)} ms`;
  }
  return `${value.toFixed(3)} s`;
}

export function formatRatio(value: number): string {
  if (!Number.isFinite(value)) {
    return "0.00x";
  }
  return `${value.toFixed(2)}x`;
}
