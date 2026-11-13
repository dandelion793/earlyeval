"""Run mini-swe-agent (gpt-5-mini, swebench config) on SWE-bench Verified tasks with online EarlyEval.

Run with the agent's environment (mini-swe-agent + lightgbm installed), e.g.
  /mnt/hdd/danhuang/early-eval-p1/.venv/bin/python earlyeval_run.py sympy__sympy-13798 [more ids] [--shadow]
      [--fail-thr 0.2] [--min-step 10] [--decide fail|paper] [--out live_online/]
Needs docker and OPENAI_API_KEY.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earlyeval_online import EarlyEvalScorer, make_earlyeval_agent  # noqa: E402

GOLD = "/mnt/hdd/danhuang/early-eval-p1/data/swebench_verified.parquet"


def run(iid, inst, scorer, a, out):
    from minisweagent.config import builtin_config_dir
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment
    cfg = yaml.safe_load((builtin_config_dir / "benchmarks" / "swebench.yaml").read_text())
    cfg["model"]["model_name"] = a.model_name
    env = get_sb_environment(cfg, inst)
    try:
        agent = make_earlyeval_agent()(
            get_model(config=cfg["model"]), env, scorer=scorer, instance_id=iid, stop=not a.shadow,
            on_step=lambda s, p, d, _: print(f"[earlyeval] {iid} step {s}: P(success)={p:.3f}"
                                             + (f"  -> {d}" if d else ""), flush=True),
            **cfg["agent"])
        info = agent.run(inst["problem_statement"])
    finally:
        env.cleanup()
    rec = {"instance_id": iid, "exit_status": info.get("exit_status"), "steps": agent.n_calls,
           "halt_step": agent.halt_step, "decision": agent.halt_decision, "ee_log": agent.ee_log,
           "submission": info.get("submission") or ""}
    (out / f"{iid}.traj.json").write_text(json.dumps({"instance_id": iid, "messages": agent.messages, "info": info},
                                                     default=str))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--shadow", action="store_true", help="score every step but never stop")
    ap.add_argument("--decide", default="fail", choices=["fail", "paper"])
    ap.add_argument("--fail-thr", type=float, default=0.2)
    ap.add_argument("--min-step", type=int, default=10)
    ap.add_argument("--model-name", default="openai/gpt-5-mini")
    ap.add_argument("--out", default="live_online")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gold = pd.read_parquet(GOLD).set_index("instance_id")
    scorer = EarlyEvalScorer(decide=a.decide, fail_thr=a.fail_thr, min_step=a.min_step)
    with ThreadPoolExecutor(len(a.ids)) as ex:
        recs = list(ex.map(lambda i: run(i, dict(gold.loc[i].to_dict(), instance_id=i), scorer, a, out), a.ids))
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    for r in recs:
        print(f"{r['instance_id']}: exit {r['exit_status']}  steps {r['steps']}  halt_step {r['halt_step']}")


if __name__ == "__main__":
    main()
