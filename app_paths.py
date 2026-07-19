from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent


def asset_path(filename):
    path = Path(filename)
    if path.name != filename:
        raise ValueError("Asset filename must not contain a path")
    return APP_ROOT / path
