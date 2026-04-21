# ACS Planner for nuPlan DB

## 0. 本次修订说明

这版代码对照论文 method/appendix 与 `implementation_notes.md` 做了几项关键修正：

- online planner 与 teacher 现在分别使用 **online / teacher** 两套 support compiler budget，避免 teacher 错用在线 `beam_width` 和 `max_atoms_per_action`。
- conservative correction 改为按动作级 upper bound 选取保守动作，更贴近 appendix 中 `U_K(a|x)` 的执行规则。
- omission upper bound 聚合时改成使用 **omitted mass + calibrated omission term** 的组合上界，避免漏掉 omission mass 的动作级惩罚。
- `nuplan_compat.py` 在无 nuPlan 环境下会正确暴露缺失状态；预处理 / teacher 脚本会在入口处直接给出清晰报错，而不是因为 `typing.Any` 伪对象继续向下执行。

---

这份工程是一个按 `implementation_notes.md` 中的系统合同落地的 **PyTorch + nuPlan DB** 版本实现，覆盖了：

- nuPlan `AbstractPlanner` wrapper
- action library / refiner
- action-conditioned support compiler
- support mass / omission damage / retained atom conditional-cost 三头网络
- teacher target generation
- residual bank 构建
- conformal omission calibration
- conservative correction / certification
- 基于 nuPlan `.db` 的预处理、训练、运行脚本

> 说明：该实现重点是把论文里的 method/runtime contract 工程化成**一套完整代码结构 + 可执行脚本**。其中 nuPlan 地图 API、simulation hydra 配置在不同 devkit 版本上可能有轻微差异，所以 README 里同时给出“直接命令”和“如果本地 devkit 配置名不同应如何替换”的说明。

---

## 1. 目录结构

```text
acs_planner_nuplan_impl_complete/
├── planner/
│   ├── nuplan_planner.py
│   ├── runtime/
│   ├── actions/
│   ├── support/
│   ├── models/
│   ├── evaluation/
│   ├── teacher/
│   ├── training/
│   └── configs/
├── scripts/
├── requirements.txt
├── setup.py
└── README.md
```

---

## 2. 你的数据路径（已按你提供的目录写入示例命令）

- nuPlan DB root:
  - `/data0/senzeyu2/dataset/nuplan/data/cache`
- train splits:
  - `/data0/senzeyu2/dataset/nuplan/data/cache/train_boston`
  - `/data0/senzeyu2/dataset/nuplan/data/cache/train_vegas_1`
  - `/data0/senzeyu2/dataset/nuplan/data/cache/train_pittsburgh`
  - `/data0/senzeyu2/dataset/nuplan/data/cache/train_singapore`
- val split:
  - `/data0/senzeyu2/dataset/nuplan/data/cache/val`
- maps root:
  - `/data0/senzeyu2/dataset/nuplan/maps`

如果你的 map version 自动识别失败，可以手动指定，例如：

```bash
--map-version nuplan-maps-v1.0
```

请根据 `maps/` 目录下真实存在的 `nuplan-maps-v*.json` 文件名调整。

---

## 3. 环境准备

### 3.1 建议环境

```bash
conda create -n acs_planner python=3.9 -y
conda activate acs_planner
```

### 3.2 安装 nuPlan devkit

如果你已经有可用的 nuPlan devkit 环境，可以跳过这一步。

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .
```

### 3.3 安装本工程依赖

```bash
cd /path/to/acs_planner_nuplan_impl_complete
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$(pwd):$PYTHONPATH
```

如果你希望每次都生效，可以把 `export PYTHONPATH=...` 写进 shell rc 文件。

---

## 4. 数据预处理：从 `.db` 生成训练张量集

该步骤会：

1. 读取 nuPlan `.db` 场景
2. 采样 scene prefix
3. 构造 runtime context
4. teacher 跑 action library + support compiler + teacher posterior
5. 导出 `(x, a, τ)` 训练样本到 `*.pt`

### 4.1 全量预处理命令

```bash
cd /path/to/acs_planner_nuplan_impl_complete
export PYTHONPATH=$(pwd):$PYTHONPATH

python scripts/preprocess_nuplan_db.py \
  --config planner/configs/default.yaml \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --maps-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-root /data0/senzeyu2/dataset/nuplan/data/cache/tensor_dataset \
  --train-splits train_boston train_vegas_2 train_pittsburgh train_singapore \
  --val-splits val
```

### 4.2 先做小规模 smoke test

推荐先跑一个很小的子集确认环境没问题：

```bash
python scripts/preprocess_nuplan_db.py \
  --config planner/configs/default.yaml \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --maps-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-root ./cache/tensor_dataset_smoke \
  --train-splits train_boston \
  --val-splits val \
  --max-train-scenarios 8 \
  --max-val-scenarios 4 \
  --max-prefixes-per-scenario 8
```

### 4.3 预处理输出

输出目录大致如下：

```text
cache/tensor_dataset/
├── train/
│   ├── index.pkl
│   ├── <scenario>_<iter>_<action>_<atom>.pt
│   └── ...
└── val/
    ├── index.pkl
    ├── <scenario>_<iter>_<action>_<atom>.pt
    └── ...
```

---

## 5. 生成 teacher cache（可选，但推荐）

这个脚本会额外把每个 prefix 的 teacher 动作结果、anchor、atom、posterior、omission target 以 pickle 形式导出来，便于做调试和可视化。

```bash
python scripts/run_teacher.py \
  --config planner/configs/teacher.yaml \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --maps-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-root ./cache/teacher_debug \
  --splits val \
  --max-scenarios 64
```

---

## 6. 构建 shared residual bank

该脚本会从预处理后的训练样本中提取 atom-anchor 表征，做轻量 prototype 选择，保存为 residual bank。

```bash
python scripts/build_residual_bank.py \
  --config planner/configs/default.yaml \
  --dataset-root ./cache/tensor_dataset \
  --split train \
  --output ./cache/residual_bank.pkl
```

如果你只是做 smoke test，可以加：

```bash
--max-samples 2000
```

---

## 7. 训练网络

训练三头网络：

- `rho_head`
- `damage_head`
- `mu_head`

命令：

```bash
python scripts/train.py \
  --config planner/configs/default.yaml \
  --data-root ./cache/tensor_dataset \
  --output-dir ./outputs/acs_default
```

训练输出：

```text
outputs/acs_default/
├── best_model.pt
└── last_model.pt
```

---

## 8. omission conformal calibration

训练完成后，在 validation set 上拟合 omission residual 的 conformal quantile：

```bash
python scripts/calibrate_omission.py \
  --config planner/configs/default.yaml \
  --data-root ./cache/tensor_dataset \
  --checkpoint ./outputs/acs_default/best_model.pt \
  --output ./outputs/acs_default/omission_calibrator.pkl
```

---

## 9. 本地快速 sanity check

在不跑 nuPlan simulation 的情况下，先确认 planner 类能正常 import：

```bash
python - <<'PY'
from planner.nuplan_planner import ACSNuPlanPlanner
planner = ACSNuPlanPlanner(
    config_path='planner/configs/default.yaml',
    checkpoint_path=None,
    residual_bank_path=None,
    calibrator_path=None,
    device='cpu',
)
print(planner.name())
PY
```

---

## 10. 接入 nuPlan closed-loop simulation

### 10.1 把 planner hydra 配置安装到 nuPlan devkit

本工程提供了一个 hydra planner config：

```text
planner/configs/nuplan_hydra/planner/acs_planner.yaml
```

把它复制到你的 `nuplan-devkit` 配置目录：

```bash
python scripts/install_nuplan_hydra_config.py \
  --nuplan-root /path/to/nuplan-devkit
```

它会把文件复制到：

```text
/path/to/nuplan-devkit/nuplan/planning/script/config/simulation/planner/acs_planner.yaml
```

### 10.2 运行 closed-loop simulation

下面给一个最常用的运行模板。不同 nuPlan devkit 版本里 `scenario_builder`、`simulation`、`observation` 的 hydra group 名可能略有不同；如果你本地已有可跑的 `run_simulation.py` 命令，最稳妥的做法就是**保留你原本能跑的命令，只把 `planner=` 换成 `acs_planner`，并补上 checkpoint/config 路径**。

示例：

```bash
cd /path/to/nuplan-devkit
export PYTHONPATH=/path/to/acs_planner_nuplan_impl_complete:$PYTHONPATH

python nuplan/planning/script/run_simulation.py \
  planner=acs_planner \
  +planner.acs_planner.config_path=/path/to/acs_planner_nuplan_impl_complete/planner/configs/default.yaml \
  +planner.acs_planner.checkpoint_path=/path/to/acs_planner_nuplan_impl_complete/outputs/acs_default/best_model.pt \
  +planner.acs_planner.residual_bank_path=/path/to/acs_planner_nuplan_impl_complete/cache/residual_bank.pkl \
  +planner.acs_planner.calibrator_path=/path/to/acs_planner_nuplan_impl_complete/outputs/acs_default/omission_calibrator.pkl
```

如果你需要显式指定数据根目录、地图根目录和 map version，请在你当前可用的 nuPlan 命令模板里继续追加对应 override。例如：

```bash
scenario_builder.data_root=/data0/senzeyu2/dataset/nuplan/data/cache
scenario_builder.map_root=/data0/senzeyu2/dataset/nuplan/maps
scenario_builder.sensor_root=/data0/senzeyu2/dataset/nuplan/data/cache
scenario_builder.map_version=nuplan-maps-v1.0
```

如果你本地 devkit 的配置 group 名不是 `scenario_builder`，请用你当前版本的真实 group 名替换。

---

## 11. 关键配置说明

### 11.1 默认训练 / 推理配置

默认配置文件：

```text
planner/configs/default.yaml
```

Teacher 高预算配置：

```text
planner/configs/teacher.yaml
```

消融开关示例：

```text
planner/configs/ablations/no_certification.yaml
planner/configs/ablations/no_topk.yaml
```

### 11.2 你最常会改的项

- `planner.output_horizon_s`
- `support.max_atoms_per_action_online`
- `ranking.global_topk_budget_online`
- `training.batch_size`
- `training.prefix_stride_s`
- `residual.residual_bank_size_per_bucket`

---

## 12. 代码中各模块职责

### `planner/nuplan_planner.py`

nuPlan runtime wrapper，负责：

1. `PlannerInput -> RuntimeContext`
2. 生成 action library
3. coarse refine + rival screening
4. compile support
5. 调网络得到 `rho / damage / mu`
6. global top-K retention
7. certification / conservative correction
8. 输出 `InterpolatedTrajectory`

### `planner/runtime/`

- runtime context builder
- route builder
- map crop cache
- agent ranking / interaction subset 选择

### `planner/actions/`

- path templates
- speed profiles
- action library
- refinement grid search

### `planner/support/`

- anchor extraction
- variable schema
- consistency rules
- atom beam search compiler

### `planner/models/`

- shared scene-action encoder
- `rho`, `damage`, `mu`, `residual_bank` heads

### `planner/teacher/`

- high-budget teacher evaluator
- omission target generation
- residual bank utilities

### `planner/training/`

- tensor dataset
- collate
- losses
- conformal calibration
- plain PyTorch trainer

---

## 13. 常见问题

### Q1. 预处理报 nuPlan import 错误

先确认你当前 shell 里已经安装并激活了 nuPlan devkit 环境；预处理和闭环仿真依赖 nuPlan API。

### Q2. 预处理时 map version 识别不对

手动传：

```bash
--map-version nuplan-maps-v1.0
```

或者改成你本地 `maps/` 目录里真实存在的 `nuplan-maps-v*.json` 文件名。

### Q3. 直接运行 `python scripts/*.py` 找不到 `planner`

先执行：

```bash
export PYTHONPATH=/path/to/acs_planner_nuplan_impl_complete:$PYTHONPATH
```

### Q4. nuPlan run_simulation hydra override 不匹配

不同 devkit 版本的 hydra group 名略有差异。最稳妥方式：

1. 先用你自己当前能跑通的 `run_simulation.py` 命令
2. 把其中 planner 替换为 `acs_planner`
3. 再补 `planner.acs_planner.*` 的 checkpoint/config override

---

## 14. 推荐执行顺序

### smoke test

```bash
python scripts/preprocess_nuplan_db.py \
  --config planner/configs/default.yaml \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --maps-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-root ./cache/tensor_dataset_smoke \
  --train-splits train_boston \
  --val-splits val \
  --max-train-scenarios 8 \
  --max-val-scenarios 4 \
  --max-prefixes-per-scenario 8

python scripts/build_residual_bank.py \
  --config planner/configs/default.yaml \
  --dataset-root ./cache/tensor_dataset_smoke \
  --split train \
  --output ./cache/residual_bank_smoke.pkl

python scripts/train.py \
  --config planner/configs/default.yaml \
  --data-root ./cache/tensor_dataset_smoke \
  --output-dir ./outputs/acs_smoke

python scripts/calibrate_omission.py \
  --config planner/configs/default.yaml \
  --data-root ./cache/tensor_dataset_smoke \
  --checkpoint ./outputs/acs_smoke/best_model.pt \
  --output ./outputs/acs_smoke/omission_calibrator.pkl
```

### full run

```bash
python scripts/preprocess_nuplan_db.py \
  --config planner/configs/default.yaml \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --maps-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-root ./cache/tensor_dataset \
  --train-splits train_boston train_vegas_1 train_pittsburgh train_singapore \
  --val-splits val

python scripts/build_residual_bank.py \
  --config planner/configs/default.yaml \
  --dataset-root ./cache/tensor_dataset \
  --split train \
  --output ./cache/residual_bank.pkl

python scripts/train.py \
  --config planner/configs/default.yaml \
  --data-root ./cache/tensor_dataset \
  --output-dir ./outputs/acs_default

python scripts/calibrate_omission.py \
  --config planner/configs/default.yaml \
  --data-root ./cache/tensor_dataset \
  --checkpoint ./outputs/acs_default/best_model.pt \
  --output ./outputs/acs_default/omission_calibrator.pkl
```

---

## 15. 最后建议

你后续真正开始做 benchmark 时，建议按这个顺序推进：

1. 先 smoke test 预处理 + 训练是否打通
2. 再跑 closed-loop sanity simulation
3. 然后再逐步替换/精化下面几个最影响性能的部分：
   - `runtime/route_builder.py` 里 map API 的 lane graph 提取
   - `support/anchor_extractor.py` 里的 conflict / merge 几何规则
   - `teacher/teacher_runner.py` 的 posterior / omission target
   - `planner/nuplan_planner.py` 里 `mu_cert` 的 residual-MC 评估

这样最容易先闭环可跑，再继续把论文细节往上补。
