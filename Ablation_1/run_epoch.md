# 消融实验三：LLM 多轮迭代生成设置

## 实验目的

评估不同 LLM 迭代生成轮数产生的 seed pool 对后续 latent-space GA 分子优化性能的影响。

本实验现在按公平消融口径整理：下游 GA 协议完全一致，候选 seed 的生成规模控制在同一量级；其中无多轮优化版本已重新使用 500 条 cold-start SMILES 转换 latent，不再使用旧的 166 条 `latent_0` 结果。

## 固定下游 GA 协议

六组实验统一使用以下 GA 参数：

```text
init_mode = llm
pop_size = 100
n_gen = 1000
elite_size = 20
cross_prob = 0.8
mut_prob = 0.08
mut_eta = 20
patience = 20
seed = 42
random_immigrant_frac = 0.05
archive_topk_per_gen = 10
max_archive_decode = 3000
final_decode_temperature = 0.8
GPU = physical GPU 0
```

每组只改变 `--llm_latent_path` 和 `--version`。

## Seed 与 Latent 文件

| LLM 设置 | SMILES 源文件 | raw SMILES | encoded latent | latent 文件 |
|---|---|---:|---:|---|
| No-multiround | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/test/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260601_145931.smi` | 500 | 500 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_no_multiround_500/llm_init_latent.npy` |
| 10 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260528_161821.smi` | 401 | 398 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_10/llm_init_latent.npy` |
| 20 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260528_162243.smi` | 288 | 288 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_20/llm_init_latent.npy` |
| 40 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260528_162837.smi` | 359 | 359 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_40/llm_init_latent.npy` |
| 80 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260528_164206.smi` | 383 | 383 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_80/llm_init_latent.npy` |
| 100 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260601_150659.smi` | 381 | 381 | `/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_100/llm_init_latent.npy` |

## 无多轮版本补全流程

### 1. 重新生成/选择 cold-start SMILES

无多轮优化版本使用 cold-start 生成文件：

```text
/root/autodl-tmp/sweeteners_evolve/Ablation_1/test/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260601_145931.smi
```

该文件包含 500 条 SMILES，文本去重后仍为 500 条。

### 2. SMILES 转 PS-VAE latent

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/molclr_pyg28/bin/python -u /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/extract_latent4llm.py --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt --input_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/test/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260601_145931.smi --out_dir /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_no_multiround_500 --gpu 0
```

转换结果：500/500 成功，`llm_init_latent.npy` shape = `(500, 56)`。

### 3. 使用统一 GA 参数重跑

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py --init_mode llm --pop_size 100 --n_gen 1000 --elite_size 20 --cross_prob 0.8 --mut_prob 0.08 --mut_eta 20 --patience 20 --seed 42 --version llm_no_multiround_500_std --llm_latent_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_no_multiround_500/llm_init_latent.npy --random_immigrant_frac 0.05 --archive_topk_per_gen 10 --max_archive_decode 3000 --final_decode_temperature 0.8
```

输出目录：`/root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_no_multiround_500_std`。

## 统一 GA 运行命令

### 无多轮优化版本

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py --init_mode llm --pop_size 100 --n_gen 1000 --elite_size 20 --cross_prob 0.8 --mut_prob 0.08 --mut_eta 20 --patience 20 --seed 42 --version llm_no_multiround_500_std --llm_latent_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_no_multiround_500/llm_init_latent.npy --random_immigrant_frac 0.05 --archive_topk_per_gen 10 --max_archive_decode 3000 --final_decode_temperature 0.8
```
### 10 轮迭代版本

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py --init_mode llm --pop_size 100 --n_gen 1000 --elite_size 20 --cross_prob 0.8 --mut_prob 0.08 --mut_eta 20 --patience 20 --seed 42 --version llm_epoch_10_std --llm_latent_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_10/llm_init_latent.npy --random_immigrant_frac 0.05 --archive_topk_per_gen 10 --max_archive_decode 3000 --final_decode_temperature 0.8
```
### 20 轮迭代版本

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py --init_mode llm --pop_size 100 --n_gen 1000 --elite_size 20 --cross_prob 0.8 --mut_prob 0.08 --mut_eta 20 --patience 20 --seed 42 --version llm_epoch_20_std --llm_latent_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_20/llm_init_latent.npy --random_immigrant_frac 0.05 --archive_topk_per_gen 10 --max_archive_decode 3000 --final_decode_temperature 0.8
```
### 40 轮迭代版本

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py --init_mode llm --pop_size 100 --n_gen 1000 --elite_size 20 --cross_prob 0.8 --mut_prob 0.08 --mut_eta 20 --patience 20 --seed 42 --version llm_epoch_40_std --llm_latent_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_40/llm_init_latent.npy --random_immigrant_frac 0.05 --archive_topk_per_gen 10 --max_archive_decode 3000 --final_decode_temperature 0.8
```
### 80 轮迭代版本

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py --init_mode llm --pop_size 100 --n_gen 1000 --elite_size 20 --cross_prob 0.8 --mut_prob 0.08 --mut_eta 20 --patience 20 --seed 42 --version llm_epoch_80_std --llm_latent_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_80/llm_init_latent.npy --random_immigrant_frac 0.05 --archive_topk_per_gen 10 --max_archive_decode 3000 --final_decode_temperature 0.8
```
### 100 轮迭代版本

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_multiple_epochs.py --init_mode llm --pop_size 100 --n_gen 1000 --elite_size 20 --cross_prob 0.8 --mut_prob 0.08 --mut_eta 20 --patience 20 --seed 42 --version llm_epoch_100_std --llm_latent_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_epoch_100/llm_init_latent.npy --random_immigrant_frac 0.05 --archive_topk_per_gen 10 --max_archive_decode 3000 --final_decode_temperature 0.8
```

## 论文建议指标

建议在表格中报告：

| LLM 设置 | Encoded latent | Best Gap ↓ | Top-10 Gap ↓ | Avg Gap ↓ | Eval@0.15 ↓ | Success Rate ↑ | Validity ↑ | Diversity ↑ | Unique SMILES ↑ | Raw Unique SMILES |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-multiround | 500 | 0.023220 | 0.029005 | 0.071511 | 4100 | 1.0000 | 1.0000 | 0.8368 | 100 | 8 |
| 10 | 398 | 0.025929 | 0.035926 | 0.065254 | 3800 | 1.0000 | 1.0000 | 0.7802 | 33 | 7 |
| 20 | 288 | 0.020450 | 0.021347 | 0.031071 | 5700 | 1.0000 | 1.0000 | 0.8448 | 44 | 12 |
| 40 | 359 | 0.020758 | 0.025016 | 0.046385 | 5600 | 1.0000 | 1.0000 | 0.8444 | 50 | 9 |
| 80 | 383 | 0.019351 | 0.027403 | 0.027403 | 3600 | 1.0000 | 1.0000 | 0.8776 | 7 | 9 |
| 100 | 381 | 0.019672 | 0.019828 | 0.020706 | 4900 | 1.0000 | 1.0000 | 0.6966 | 100 | 34 |

说明：旧的 `llm_epoch_1_std` 基于 166 条 latent，已不再作为正式无多轮结果使用。80 轮最终 unique population 只有 7 个分子，因此该组 `Top-10 Gap` 实际按可用的 7 个 unique 分子计算；论文图表中建议同时标注 `Unique SMILES`，避免只看 gap 造成误读。

本轮标准化结果汇总 CSV：`/root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_epoch_std_summary.csv`。

图建议：

1. Top-10 Gap 柱状图：展示不同 LLM 轮数下 top-k 优化质量。
2. Diversity / Unique SMILES 柱状图：展示多轮生成对候选多样性的影响。
3. Eval@0.15 柱状图：展示达到成功阈值所需 evaluation 数量。
