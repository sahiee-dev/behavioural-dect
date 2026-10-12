import json
import sys
from collections import Counter
from pathlib import Path

repo = Path(sys.argv[1])
cats = Counter()
for split in ["train", "val", "test"]:
    for f in (repo / "data_taskdisjoint" / split).glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("agent") != "web":
            continue
        labels = {s["label"] for s in d["steps"]}
        key = (d["category"], "hijacked" in labels)
        cats[key] += 1

for k, v in sorted(cats.items()):
    print(k, v)
