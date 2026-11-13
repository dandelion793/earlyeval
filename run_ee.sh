#!/bin/bash
cd /mnt/hdd/danhuang/earlyeval-repro
export SWE_PREFIX_SKIP_INSTANCE_DEDUP=1 OMP_NUM_THREADS=16 EARLYEVAL_VENDOR_RUNTIME_ROOT=$PWD/outputs/vendor_runtime/ppmha
P=earlyeval/vendor/prefix_predict_model_holdout_answer/run_all.py
.venv/bin/python $P --data-dir in_r1 --run-name r1_repro --split-by model_holdout --holdout-models auto_mid3 --verified-jsonl data/verified.jsonl --no-gpu-lgbm > outputs/r1_repro.log 2>&1 &
.venv/bin/python $P --data-dir in_r2 --run-name r2_ours --split-by model_holdout --holdout-models ours_f1_new30,ours_f1,ours_g1,ours_g2,ours_g3,ours_h0_new30a,ours_h0_new30b,ours_h0_pilot,ours_L1,20260217_mini-v2.0.0_gpt-5-mini --verified-jsonl data/verified.jsonl --no-gpu-lgbm --skip-ablation > outputs/r2_ours.log 2>&1 &
wait
echo done > outputs/ee_done
