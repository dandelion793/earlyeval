"""Score mini-swe-agent trajectories with the reproduced EarlyEval (R2 run: trained on 38 archive systems,
our gpt-5-mini runs held out). Uses EarlyEval's own step/prefix builders, gold-answer enrichment, saved
FeatureEngineer and LightGBM model + sigmoid calibrator; the trajectory's model id is masked as for held-out agents.

python earlyeval_infer.py --trajs DIR_WITH_.traj.json --out live5/earlyeval_scores.csv [--model H_LightGBM_Dense]
"""
import argparse, glob, json, os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
V = ROOT / "earlyeval/vendor/prefix_predict_model_holdout_answer"
sys.path.insert(0, str(V)); sys.path.insert(0, str(ROOT))
os.environ.setdefault("EARLYEVAL_VENDOR_RUNTIME_ROOT", str(ROOT / "outputs/vendor_runtime/ppmha"))
from convert_archive import to_messages  # noqa: E402

MODELS = {"H_LightGBM_Dense": ("baseline_lgbm_dense.lgb", "dense"),
          "I_LightGBM_Dense_AF": ("baseline_lgbm_dense_af.lgb", "dense_af")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=str(V / "runs/r2_ours"))
    ap.add_argument("--model", default="H_LightGBM_Dense")
    ap.add_argument("--verified", default=str(ROOT / "data/verified.jsonl"))
    a = ap.parse_args()
    from prefix_builder import build_prefix_samples_for_trajectory
    from answer_features import enrich_prefix_with_answer_features
    from feature_engineer import FeatureEngineer
    import lightgbm as lgb
    rows = []
    for f in sorted(glob.glob(f"{a.trajs}/*.traj.json")):
        t = json.load(open(f)); iid = t.get("instance_id") or Path(f).name.split(".traj")[0]
        row = pd.Series({"traj_id": f"live::{iid}", "instance_id": iid, "resolved": False, "model": "__MISSING__",
                         "messages": json.dumps(to_messages(t))})
        rows += build_prefix_samples_for_trajectory(row)
    df = pd.DataFrame(rows)
    df, _ = enrich_prefix_with_answer_features(df, a.verified)
    for c, v in (("orig_model_id", "live"), ("orig_model", "live")):
        df[c] = v
    if "model_id" in df: df["model_id"] = "__MISSING__"
    df["model"] = "__MISSING__"; df["model_id_input_mode"] = "heldout"
    run = Path(a.run) / "models"
    fe = FeatureEngineer.load(run / "feature_engineer_with_model.pkl")
    fname, kind = MODELS[a.model]
    X = fe.transform_dense(df)
    if kind == "dense_af":
        from scipy import sparse
        cols = [c for c in fe.active_text_columns if c in ("action_text_tfidf", "feedback_text_tfidf")] or list(fe.active_text_columns)
        X = sparse.hstack([sparse.csr_matrix(X)] + [fe.transform_tfidf_column(df, c) for c in cols]).tocsr()
    booster = lgb.Booster(model_file=str(run / fname))
    raw = booster.predict(X)
    cal = pickle.load(open(run / f"calibrator_{a.model}.pkl", "rb"))
    df["p_raw"], df["p_cal"] = raw, cal.predict(raw)
    out = df[["instance_id", "prefix_step_idx", "p_raw", "p_cal"]].sort_values(["instance_id", "prefix_step_idx"])
    out.to_csv(a.out, index=False); print(out.groupby("instance_id").tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
