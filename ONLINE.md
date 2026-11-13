# EarlyEval 在线使用：边做任务边打分，决定是否停下

coding agent 每走一步，就用复现的 EarlyEval 计算一次 P(success)（已校准）。满足叫停规则时，以 `exit_status = "EarlyStopFail"` 结束这次运行。

## 前提
- 用 agent 所在的 Python 环境运行（需要 mini-swe-agent 2.2.7 和 lightgbm 4.7.0）。`/mnt/hdd/danhuang/early-eval-p1/.venv` 已经装好。
- 模型：`earlyeval/vendor/prefix_predict_model_holdout_answer/runs/r2_ours/` 下的
  - `feature_engineer_with_model.pkl`
  - `baseline_lgbm_dense.lgb`
  - `calibrator_H_LightGBM_Dense.pkl`
- 题目和官方答案：`data/verified.jsonl`。
- **只能用于 SWE-bench Verified 的题目，也只适合"省评测成本"这个场景。** 这个 EarlyEval 模型和论文一样用了官方答案特征（比如答案改了哪些文件、agent 碰过的文件和答案有多少重合），真实部署时拿不到这些信息。不用官方答案的早评，请用 `early-eval-ourmethod/earlystop`。

## 命令行：直接跑任务

```bash
cd /mnt/hdd/danhuang/earlyeval-repro
set -a; . ~/.config/mini-swe-agent/.env; set +a                       # 读入 OPENAI_API_KEY
PY=/mnt/hdd/danhuang/early-eval-p1/.venv/bin/python

$PY earlyeval_run.py sympy__sympy-13798 scikit-learn__scikit-learn-12585 --out live_online   # 叫停模式
$PY earlyeval_run.py django__django-11999 --shadow                     # 影子模式：只打分不停
$PY earlyeval_run.py <ids> --fail-thr 0.3 --min-step 10                # 调整叫停阈值
```

输出：
- 终端会逐步打印 `[earlyeval] <id> step s: P(success)=...`，触发叫停时附带 `-> fail`；
- `--out` 目录下每道题有一份 `<id>.traj.json`，另有汇总文件 `results.jsonl`（结束状态、叫停步数、每步概率、补丁）。

## 在自己的代码里调用

```python
import sys; sys.path.insert(0, "/mnt/hdd/danhuang/earlyeval-repro")
from earlyeval_online import EarlyEvalScorer, make_earlyeval_agent

scorer = EarlyEvalScorer(decide="fail", fail_thr=0.2, min_step=10)   # 加载模型约 0.5 秒

# A) 接入 mini-swe-agent
Agent = make_earlyeval_agent()                      # DefaultAgent 子类；也可以传入你自己的类：make_earlyeval_agent(MyAgent)
agent = Agent(model, env, scorer=scorer, instance_id=iid,
              stop=True,                            # False = 影子模式
              on_step=lambda s, p, d, a: print(s, p, d),   # 每步回调：步数、P(success)、判定（None/"fail"/"success"）
              **cfg["agent"])
info = agent.run(task)
info["exit_status"]   # "EarlyStopFail"（已叫停）或 "Submitted"
agent.ee_log          # 每步的 P(success)
agent.halt_step       # 叫停的步数

# B) 自己的 agent 循环里手动调用
step, p = scorer.prob(messages, iid)                # 最新一步之后的 P(success)
if scorer.decision(step, p) == "fail":
    ...                                             # 停下，或者交给后续处理（诊断、修改 harness、重跑）
```

## 叫停规则

| `decide` | 规则 | 说明 |
|---|---|---|
| `"fail"`（默认） | 第 `min_step`（10）步起，P ≤ `fail_thr`（0.2）就停 | 在我们 118 次原 harness 运行上：叫停 16 次，全部确实失败 |
| `"paper"` | P ≥ 0.95 判成功，P ≤ 0.05 判失败 | 论文的双阈值规则。默认只有判失败时才停；`stop_on_success=True` 时判成功也停，但那样运行就没有补丁了，只适合用预测结果代替真实结果来估算分数 |

## 实测（2026-11-13，gpt-5-mini，叫停模式）

| 题目 | 每步 P(success) | 结果 |
|---|---|---|
| sympy-13798 | 0.09, 0.10, 0.07, …, 0.06, **0.07（第 10 步）** | 第 10 步 `EarlyStopFail`。之前完整跑过一次，那次确实失败，共 23 步，等于省下 13 步 |
| scikit-learn-12585 | 0.87–0.91，一直很高 | 不叫停，第 13 步正常提交。之前原 harness 两次都做对 |

- **速度**：每步打分约 0.24 秒（每步都会重建全部前缀，并重新读一次官方答案文件）。相比 agent 每步调用 LLM 的几秒，可以忽略不计。
- **一致性**：在线逐步算出的分数，和离线脚本 `earlyeval_infer.py` 对整条轨迹算出的分数完全一致（最大差 1e-16）。离线结果和 R2 训练时存下的预测也一致（最大差 3e-8）。

## 局限
它和我们的早评一样，主要依据题目难度，同一道题内分不出这次运行是否跑偏。例如 django-14007 题目先验高，这次运行失败了，它没有叫停。详见 `live5/README.md`。
