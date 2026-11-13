# EarlyEval（复现版）在 5 道题上的测试

## 用的模型
R2 训练的 EarlyEval：
- 训练数据：38 个历史 mini-swe-agent 系统；我们自己的 gpt-5-mini 运行全部留作测试，没有参与训练。
- 用 EarlyEval 自己的代码处理轨迹：逐步表、前缀样本、官方答案特征、保存的 FeatureEngineer。
- 模型：LightGBM `H_LightGBM_Dense`，加 sigmoid 校准器。
- 模型标识按留出 agent 的方式遮掉（`__MISSING__`）。

## 用的数据
用的是 `early-eval-ourmethod/results/earlystop_live/` 里那 5 次真实运行：gpt-5-mini 加原版 mini-swe-agent，完整跑完，结果经 SWE-bench 官方评测。
- `trajs/` 是软链接，指向那份轨迹；
- 真实结果在那里的 `results.jsonl`。

## 步骤

```bash
cd /mnt/hdd/danhuang/earlyeval-repro
# 1) 给每条轨迹逐步打分（P(success)，有原始值 p_raw 和校准后 p_cal）
.venv/bin/python earlyeval_infer.py --trajs live5/trajs --out live5/earlyeval_scores.csv [--model H_LightGBM_Dense]
# 2) 校验推理代码：对一条 R2 已经评过的运行，结果应和 R2 存下的预测一致（实测最大差 3e-8）
.venv/bin/python earlyeval_infer.py --trajs live5/check --out live5/check.csv
```

用到的模型文件都在 `earlyeval/vendor/prefix_predict_model_holdout_answer/runs/r2_ours/models/`：
- `feature_engineer_with_model.pkl`
- `baseline_lgbm_dense.lgb`
- `calibrator_H_LightGBM_Dense.pkl`

## 叫停规则
| 规则 | 内容 |
|---|---|
| 论文规则 | 第一次出现 p_cal ≥ 0.95（判成功）或 ≤ 0.05（判失败）时停 |
| 只判失败，阈值 0.2 | 第 10 步起 p_cal ≤ 0.2 就停（在我们 118 次原 harness 运行上精度为 100%） |
| 只判失败，阈值 0.41 | 和我们早评用同一个阈值 |

## 结果（2026-11-13）

| 题目 | 题目先验 | 第 10 步 P | 最后一步 P | 论文规则 | 阈值 0.2 | 阈值 0.41 | 真实结果 | 我们的早评 |
|---|---|---|---|---|---|---|---|---|
| django-14007 | 0.86 | 0.66 | 0.61 | 不决定 | 不停 | 不停 | 失败 | 不停 |
| django-14771 | 0.56 | 0.65 | 0.54 | 不决定 | 不停 | 不停 | 成功 | 不停 |
| matplotlib-24570 | 0.78 | 0.82 | 0.79 | 不决定 | 不停 | 不停 | 成功 | 不停 |
| scikit-learn-12585 | 0.94 | 0.79 | 0.71 | 不决定 | 不停 | 不停 | 成功 | 不停 |
| sympy-13798 | 0.03 | 0.07 | 0.08 | 不决定 | 第 10 步停 | 第 10 步停 | 失败 | 第 10 步停 |
