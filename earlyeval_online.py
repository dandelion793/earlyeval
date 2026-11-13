"""Online EarlyEval: score a running mini-swe-agent after every step and decide whether to stop it.

    import sys; sys.path.insert(0, "/mnt/hdd/danhuang/earlyeval-repro")
    from earlyeval_online import EarlyEvalScorer, make_earlyeval_agent
    scorer = EarlyEvalScorer()                                 # loads the R2 model (once, ~10 s)
    p = scorer.prob(agent.messages, instance_id)               # calibrated P(success) after the latest step
    Agent = make_earlyeval_agent()                             # mini-swe-agent DefaultAgent subclass
    agent = Agent(model, env, scorer=scorer, instance_id=iid, stop=True, **cfg["agent"])

Uses EarlyEval's own feature pipeline (step/prefix builders, gold-answer enrichment, saved FeatureEngineer,
LightGBM + sigmoid calibrator from runs/r2_ours). The agent's model id is masked as for held-out agents.

NOTE: this EarlyEval model includes reference-solution (gold patch) features, as in the paper. It can only be
used on SWE-bench Verified tasks, for saving evaluation cost; not in a deployment where the fix is unknown.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "earlyeval/vendor/prefix_predict_model_holdout_answer"
DEFAULT_RUN = VENDOR / "runs/r2_ours"
os.environ.setdefault("EARLYEVAL_VENDOR_RUNTIME_ROOT", str(ROOT / "outputs/vendor_runtime/ppmha"))
for p in (str(VENDOR), str(ROOT)):
    if p not in sys.path:
        sys.path.append(p)


def _steps_of(messages):
    """Normalise mini-swe-agent v1/v2 messages to EarlyEval's raw format (assistant action + tool output)."""
    from convert_archive import to_messages
    return to_messages({"messages": messages})


class EarlyEvalScorer:
    """Calibrated P(success | first s steps) from the reproduced EarlyEval.

    Stop rules (decide=):
      - "fail":  stop at the first step >= min_step with P <= fail_thr   (default 0.2: 100% precision on our
                 118 original-harness runs, 16 stops)
      - "paper": stop at the first step with P >= 1 - paper_thr (predict success) or P <= paper_thr (fail);
                 paper_thr = 0.05 is the paper's 0.95 confidence
    """

    def __init__(self, run_dir: str | Path = DEFAULT_RUN, model: str = "H_LightGBM_Dense",
                 verified_jsonl: str | Path = ROOT / "data/verified.jsonl", decide: str = "fail",
                 fail_thr: float = 0.2, min_step: int = 10, paper_thr: float = 0.05):
        import lightgbm as lgb
        from feature_engineer import FeatureEngineer
        models = {"H_LightGBM_Dense": "baseline_lgbm_dense.lgb"}
        if model not in models:
            raise ValueError(f"supported models: {list(models)}")
        run = Path(run_dir) / "models"
        self.fe = FeatureEngineer.load(run / "feature_engineer_with_model.pkl")
        self.booster = lgb.Booster(model_file=str(run / models[model]))
        self.cal = pickle.load(open(run / f"calibrator_{model}.pkl", "rb"))
        self.verified = str(verified_jsonl)
        self.decide, self.fail_thr, self.min_step, self.paper_thr = decide, fail_thr, min_step, paper_thr

    def _frame(self, messages, instance_id):
        from answer_features import enrich_prefix_with_answer_features
        from prefix_builder import build_prefix_samples_for_trajectory
        row = pd.Series({"traj_id": f"live::{instance_id}", "instance_id": instance_id, "resolved": False,
                         "model": "__MISSING__", "messages": json.dumps(_steps_of(messages))})
        df = pd.DataFrame(build_prefix_samples_for_trajectory(row))
        if df.empty:
            return df
        df, _ = enrich_prefix_with_answer_features(df, self.verified)
        if "model_id" in df:
            df["model_id"] = "__MISSING__"
        df["model"], df["model_id_input_mode"] = "__MISSING__", "heldout"
        df["orig_model_id"] = df["orig_model"] = "live"
        return df

    def probs(self, messages, instance_id) -> list[tuple[int, float]]:
        """[(step, calibrated P(success))] for every prefix of the trajectory so far."""
        df = self._frame(messages, instance_id)
        if df.empty:
            return []
        raw = self.booster.predict(self.fe.transform_dense(df))
        cal = self.cal.predict(np.asarray(raw))
        return list(zip(df["prefix_step_idx"].astype(int).tolist(), [float(x) for x in cal]))

    def prob(self, messages, instance_id) -> tuple[int, float] | None:
        """(step, calibrated P(success)) after the latest step."""
        ps = self.probs(messages, instance_id)
        return max(ps) if ps else None

    def decision(self, step: int, p: float) -> str | None:
        """'fail' / 'success' if the rule fires at this step, else None."""
        if self.decide == "paper":
            if p >= 1 - self.paper_thr:
                return "success"
            if p <= self.paper_thr:
                return "fail"
            return None
        return "fail" if step >= self.min_step and p <= self.fail_thr else None


def make_earlyeval_agent(base=None):
    """mini-swe-agent agent that scores every step with EarlyEval; stop=True ends the run when the rule fires
    (exit_status 'EarlyStopFail' or 'EarlyStopSuccess'); stop=False only logs (shadow mode)."""
    from minisweagent.exceptions import InterruptAgentFlow
    if base is None:
        from minisweagent.agents.default import DefaultAgent as base

    class EarlyEvalAgent(base):
        def __init__(self, *a, scorer: EarlyEvalScorer, instance_id: str, stop: bool = True, on_step=None,
                     stop_on_success: bool = False, **kw):
            """stop_on_success=True also ends runs predicted to succeed (only meaningful when the prediction
            replaces the real outcome, e.g. estimating a benchmark score; the run then has no patch)."""
            super().__init__(*a, **kw)
            self.scorer, self.instance_id, self.stop, self.on_step = scorer, instance_id, stop, on_step
            self.stop_on_success = stop_on_success
            self.ee_log: list[float] = []
            self.halt_step, self.halt_decision = None, None

        def step(self):
            out = super().step()
            sp = self.scorer.prob(self.messages, self.instance_id)
            if sp is None:
                return out
            s, p = sp
            self.ee_log.append(p)
            d = self.scorer.decision(s, p)
            if self.on_step:
                self.on_step(s, p, d, self)
            if self.stop and d and self.halt_step is None and (d == "fail" or self.stop_on_success):
                self.halt_step, self.halt_decision = s, d
                status = "EarlyStopFail" if d == "fail" else "EarlyStopSuccess"
                raise InterruptAgentFlow({"role": "exit", "content": status,
                                          "extra": {"exit_status": status, "submission": ""}})
            return out

    return EarlyEvalAgent
