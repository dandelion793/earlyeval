"""Our own gpt-5-mini runs in EarlyEval raw format (held-out test agents)."""
import glob, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/mnt/hdd/danhuang/early-eval-p1")
from p2 import online as O
from convert_archive import to_messages
def res(a):
    out = {}
    for f in glob.glob(f"{O.OUT}/logs/run_evaluation/online-{a}/online-{a}/*/report.json"):
        d = json.load(open(f)); k = list(d)[0]; out[k] = d[k]["resolved"]
    return out
GROUPS = {"ours_h0_pilot": ["A", "test0"], "ours_h0_new30a": ["n_h0"], "ours_h0_new30b": ["n_h0b"],
          "ours_g1": ["g1"], "ours_g2": ["g2"], "ours_g3": ["g3"], "ours_L1": ["L1"], "ours_f1": ["f1"], "ours_f1_new30": ["n_f1"]}
for name, arms in GROUPS.items():
    rows = []
    for a in arms:
        R = res(a)
        for p in glob.glob(str(O.RUNS / a / "*" / "attempt1.traj.json")):
            iid = Path(p).parent.name
            t = json.loads(open(p).read()); m = to_messages(t)
            if len(m) < 3: continue
            rows.append({"traj_id": f"{name}::{iid}", "instance_id": iid, "resolved": bool(R.get(iid, False)), "model": name,
                         "messages": json.dumps(m)})
    pd.DataFrame(rows).to_parquet(f"data/ours/tool-{name}.parquet")
    print(name, len(rows), sum(r["resolved"] for r in rows))
