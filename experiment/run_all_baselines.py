import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.common.runner import evaluate_method, format_results


METHODS = [
    ("largest_empty_circle", "experiment.largest_empty_circle.method"),
    ("rotated_rectangle", "experiment.rotated_rectangle.method"),
    ("heuristic_ellipse_fit", "experiment.heuristic_ellipse_fit.method"),
    ("segmentation_polygon_postprocess", "experiment.segmentation_polygon_postprocess.method"),
    ("direct_polygon_regression", "experiment.direct_polygon_regression.method"),
    ("iris", "experiment.iris.method"),
    ("decomputil", "experiment.decomputil.method"),
    ("firi", "experiment.firi.method"),
]


def main(max_samples: int = 300):
    rows = []
    skipped = []

    for name, module_path in METHODS:
        print(f"\\n[Running] {name}")
        try:
            result = evaluate_method(name, module_path, max_samples=max_samples)
            rows.append(result)
            print("done")
        except NotImplementedError as e:
            skipped.append((name, str(e)))
            print(f"skip: {e}")
        except Exception as e:
            skipped.append((name, f"unexpected error: {e}"))
            print("error")
            traceback.print_exc()

    print("\\n=== Baseline Result Summary ===")
    if rows:
        print(format_results(rows))
    else:
        print("No completed methods.")

    if skipped:
        print("\\n=== Skipped Methods ===")
        for name, reason in skipped:
            print(f"- {name}: {reason}")


if __name__ == "__main__":
    main(max_samples=300)
