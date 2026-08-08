import json
import os


_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.json")
)

_VALID_SOLVERS = {"hpipm", "osqp"}
_VALID_INTERFACES = {"python", "cpp"}


def load_project_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_vehicle_parameters():
    return load_project_config().get("vehicle_parameters", {})


def get_planner_settings():
    config = load_project_config()
    planner_settings = config.get("planner_settings", {})

    solver = str(
        planner_settings.get("solver", planner_settings.get("planner_mode", "osqp"))
    ).lower()
    neural_iris_interface = str(
        planner_settings.get(
            "neural_iris_interface",
            planner_settings.get("interface", planner_settings.get("backend", "cpp")),
        )
    ).lower()

    if solver not in _VALID_SOLVERS:
        raise ValueError(
            f"Unsupported planner solver in config: {solver}. "
            f"Supported values: {sorted(_VALID_SOLVERS)}"
        )

    if neural_iris_interface not in _VALID_INTERFACES:
        raise ValueError(
            f"Unsupported planner interface in config: {neural_iris_interface}. "
            f"Supported values: {sorted(_VALID_INTERFACES)}"
        )

    return {
        "solver": solver,
        "neural_iris_interface": neural_iris_interface,
    }
