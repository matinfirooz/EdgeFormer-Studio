import csv
import json
import platform
from pathlib import Path
import torch
from edgeformer import __version__


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment(device="cpu"):
    return dict(
        edgeformer=__version__,
        python=platform.python_version(),
        torch=str(torch.__version__),
        platform=platform.platform(),
        device=str(device),
        threads=torch.get_num_threads(),
        cuda=torch.version.cuda,
        device_name=torch.cuda.get_device_name(device)
        if str(device).startswith("cuda")
        else platform.processor(),
    )
