import os
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--reference', action='store_true')
args = parser.parse_args()
os.environ["LEAKAGELAB_INPUT"] = str(root / 'artifacts/local/data/solver')
source = 'reference-solver.ipynb' if args.reference else 'leakagelab-kaggle.ipynb'
notebook = nbformat.read(root / 'notebooks' / source, as_version=4)
NotebookClient(notebook, timeout=1800, kernel_name='python3', resources={"metadata":{"path":str(root)}}).execute()
output = root / ('artifacts/reference-solver-local-executed.ipynb' if args.reference else 'artifacts/leakagelab-local-executed.ipynb')
nbformat.write(notebook, output)
print('Executed clean-state notebook:', output)
