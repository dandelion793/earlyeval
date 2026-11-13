#!/bin/bash
cd /mnt/hdd/danhuang/earlyeval-repro
export SWE_PREFIX_SKIP_INSTANCE_DEDUP=1 OMP_NUM_THREADS=24 EARLYEVAL_VENDOR_RUNTIME_ROOT=$PWD/outputs/vendor_runtime/ppmha
.venv/bin/python earlyeval/vendor/prefix_predict_model_holdout_answer/run_all.py --data-dir in_r2 --run-name r2_ours --split-by model_holdout --holdout-models ours_f1_new30,ours_f1,ours_g1,ours_g2,ours_g3,ours_h0_new30a,ours_h0_new30b,ours_h0_pilot,ours_L1,20260217_mini-v2.0.0_gpt-5-mini --verified-jsonl data/verified.jsonl --no-gpu-lgbm --skip-ablation --skip-step-table --skip-prefix-table > outputs/r2_ours_b.log 2>&1
echo $? > outputs/r2_done
