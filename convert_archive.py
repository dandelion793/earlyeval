"""Re-download the full mini-SWE-agent SWE-bench Verified trajectories (public S3 submissions of the
SWE-bench/experiments repo, the same systems as our compact archive) and write them in EarlyEval's raw
format: data/swe/tool-<system>.parquet with columns traj_id, instance_id, resolved, model, messages (JSON).
Labels come from our compact archive (already resolved per system/instance)."""
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, "/mnt/hdd/danhuang/early-eval-p1")
from p1.download import EXPERIMENTS_RAW, get, s3_list, split_s3  # noqa: E402
from p2.llm_eval import steps_of  # noqa: E402

COMPACT = Path("/mnt/hdd/danhuang/early-eval-p1/data/compact")
OUT = Path("/mnt/hdd/danhuang/earlyeval-repro/data/swe")
OUT.mkdir(parents=True, exist_ok=True)
OUT_CH, THOUGHT_CH = 6000, 4000


def clip(s, n):
    s = s or ""
    return s if len(s) <= n else s[: n * 2 // 3] + "\n...\n" + s[-n // 3:]


def to_messages(traj: dict) -> list[dict]:
    msgs = traj.get("messages") or traj.get("trajectory") or []
    out = [{"role": "user", "content": "task"}]
    for s in steps_of(msgs):
        cmd = "\n".join(s["commands"])
        out.append({"role": "assistant", "message_type": "action", "thought": clip(s["thought"], THOUGHT_CH),
                    "action": cmd, "content": clip(s["thought"], THOUGHT_CH) + f"\n```bash\n{cmd}\n```",
                    "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]})
        out.append({"role": "tool", "message_type": "observation", "content": clip(s["output"], OUT_CH)})
    return out


def system(name: str):
    out = OUT / f"tool-{name}.parquet"
    if out.exists():
        print(f"[skip] {name}", flush=True)
        return
    labels = {}
    with gzip.open(COMPACT / f"{name}.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            labels[r["instance_id"]] = str(r["resolved"]) == "True"
    meta = yaml.safe_load(get(f"{EXPERIMENTS_RAW}/{name}/metadata.yaml") or b"{}") or {}
    bucket, prefix = split_s3(meta["assets"]["trajs"])
    keys = [k for k in s3_list(bucket, prefix) if k.endswith(".traj.json")]

    def one(key):
        raw = get(f"https://{bucket}.s3.amazonaws.com/{key}")
        if not raw:
            return None
        try:
            t = json.loads(raw)
            iid = (t.get("instance_id") or (t.get("info") or {}).get("instance_id") or key.rsplit("/", 2)[-2])
            m = to_messages(t)
        except Exception:  # noqa: BLE001
            return None
        if iid not in labels or len(m) < 3:
            return None
        return {"traj_id": f"{name}::{iid}", "instance_id": iid, "resolved": labels[iid], "model": name,
                "messages": json.dumps(m)}

    with ThreadPoolExecutor(24) as ex:
        rows = [r for r in ex.map(one, keys) if r]
    pd.DataFrame(rows).to_parquet(out)
    print(f"[ok] {name}: {len(rows)}/{len(keys)}", flush=True)


if __name__ == "__main__":
    names = sorted(p.name[:-9] for p in COMPACT.glob("*.jsonl.gz"))
    for n in names:
        try:
            system(n)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {n}: {type(e).__name__}: {e}", flush=True)
