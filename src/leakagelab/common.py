from pathlib import Path
import hashlib
import json
import yaml


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(df):
    return hashlib.sha256(df.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()).hexdigest()
