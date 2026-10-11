"""Package the canonical .ipynb and a separate, reproducible source input."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import nbformat as nb

root = Path(__file__).resolve().parent
files = [p for pattern in ["src/leakagelab/*.py", "tests/*.py", "configs/*.yaml", "pyproject.toml"]
         for p in sorted(root.glob(pattern))]
notebook = nb.read(root / "notebooks/leakagelab-kaggle.ipynb", as_version=4)
nb.validate(notebook)
(root / "kaggle").mkdir(exist_ok=True)
nb.write(notebook, root / "kaggle/leakagelab-kaggle.ipynb")
reference = nb.read(root / "notebooks/reference-solver.ipynb", as_version=4)
for cell in reference.cells:
    if cell.cell_type == "code" and cell.source.startswith("%%writefile "):
        path = cell.source.splitlines()[0].removeprefix("%%writefile ")
        cell.source = f"%%writefile {path}\n" + (root / path).read_text(encoding="utf-8")
nb.validate(reference)
nb.write(reference, root / "notebooks/reference-solver.ipynb")
with ZipFile(root / "artifacts/leakagelab-source.zip", "w", ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root).as_posix())
metadata_path = root / "kaggle/kernel-metadata.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_sources"] = ["vanh1803/leakagelab-source"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Validated canonical notebooks; packaged {len(files)} source/config/test files.")
