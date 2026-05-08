import type {
  BenchmarkJobResponse,
  BenchmarkOptions,
  BenchmarkProgressEvent,
  BenchmarkRequest,
  DatasetSummary
} from "../types/benchmark";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL;

export const API_BASE_URL =
  configuredBaseUrl && configuredBaseUrl.length > 0
    ? configuredBaseUrl
    : window.location.port === "5173"
      ? "http://localhost:8000"
      : window.location.origin;

const websocketBaseUrl = API_BASE_URL.replace(/^http/, "ws");

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Keep the HTTP status text when the response body is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function fetchDatasetSummary(): Promise<DatasetSummary> {
  const response = await fetch(`${API_BASE_URL}/api/dataset-summary`);
  return readJson<DatasetSummary>(response);
}

export async function fetchBenchmarkOptions(): Promise<BenchmarkOptions> {
  const response = await fetch(`${API_BASE_URL}/api/benchmark/options`);
  return readJson<BenchmarkOptions>(response);
}

export async function startBenchmark(
  request: BenchmarkRequest
): Promise<BenchmarkJobResponse> {
  const response = await fetch(`${API_BASE_URL}/api/benchmark/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request)
  });
  return readJson<BenchmarkJobResponse>(response);
}

export function connectBenchmarkProgress(
  jobId: string,
  onEvent: (event: BenchmarkProgressEvent) => void,
  onError: () => void
): WebSocket {
  const socket = new WebSocket(
    `${websocketBaseUrl}/api/benchmark/${jobId}/progress`
  );

  socket.onmessage = (message) => {
    onEvent(JSON.parse(message.data) as BenchmarkProgressEvent);
  };

  socket.onerror = () => {
    onError();
  };

  return socket;
}
