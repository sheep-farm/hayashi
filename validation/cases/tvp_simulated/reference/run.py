import json
from pathlib import Path

here = Path(__file__).resolve().parent
data_dir = here.parent / "data"

with open(data_dir / "true_final.json") as f:
    result = json.load(f)

print(json.dumps(result, indent=2))
