from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sketchlinalg.benchmarks.count_sketch_benchmarks import benchmark
from sketchlinalg.datasets import load_e2006, recache_e2006


def run_e2006_benchmark(
        sketch_dims: list[int] | None = None,
        alphas: list[float] | None = None,
        *,
        repeats: int = 3,
        seed: int = 127,
        force_recache: bool = False,
) -> list[dict]:
    dataset = recache_e2006() if force_recache else load_e2006()
    sketch_dims = sketch_dims or [5000, 20000, 50000]
    alphas = alphas or [1e-6, 1.0, 5.0]

    return benchmark(
        dataset.train.X,
        dataset.train.y,
        dataset.test.X,
        dataset.test.y,
        sketch_dims,
        alphas,
        repeats=repeats,
        seed=seed,
    )


def main() -> None:
    for result in run_e2006_benchmark(force_recache=True):
        print(result)


if __name__ == "__main__":
    main()
