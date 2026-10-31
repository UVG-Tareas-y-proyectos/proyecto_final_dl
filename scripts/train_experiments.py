"""Ejecuta una matriz de entrenamientos; test se reserva para la siguiente parte."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def commands(config):
    required = {"root", "manifest", "out", "epochs", "batch_size", "lr", "weight_decay", "seed", "threads", "device", "runs"}
    if set(config) != required or not config["runs"]:
        raise ValueError("La configuración no contiene los campos esperados.")
    names = set()
    result = []
    for run in config["runs"]:
        if set(run) != {"name", "model", "class_weights"}:
            raise ValueError("Cada ejecución requiere name, model y class_weights.")
        name = run["name"]
        if not isinstance(name, str) or not name or name in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name) or name in names:
            raise ValueError("Nombre de ejecución inválido o repetido.")
        names.add(name)
        if run["model"] not in {"cnn", "mobilenet"} or run["class_weights"] not in {"none", "balanced"}:
            raise ValueError("Modelo o ponderación desconocidos.")
        cmd = [sys.executable, "-m", "plant_disease.train"]
        for key in sorted(required - {"runs", "out"}):
            cmd += ["--" + key.replace("_", "-"), str(config[key])]
        cmd += ["--out", str(Path(config["out"]) / name), "--model", run["model"], "--class-weights", run["class_weights"]]
        result.append(cmd)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_part2.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runs = commands(config)
    out = Path(config["out"])
    if out.exists() and any(out.iterdir()):
        parser.error("Usa una carpeta de salida nueva para conservar los experimentos.")
    for run, command in zip(config["runs"], runs):
        print(f"Entrenamiento: {run['name']}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
