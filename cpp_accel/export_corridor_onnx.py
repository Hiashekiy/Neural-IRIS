import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.corridor_constraints.model import CorridorEllipseNet


def main():
    model_path = os.path.join(ROOT, "models", "iris_net_best.pth")
    out_dir = os.path.join(ROOT, "cpp_accel", "models")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "corridor_ellipse_net.onnx")

    device = torch.device("cpu")
    model = CorridorEllipseNet().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 1, 128, 128, dtype=torch.float32, device=device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )

    print(f"ONNX exported to: {onnx_path}")


if __name__ == "__main__":
    main()
