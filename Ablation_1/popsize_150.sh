#!/bin/bash

数据处理环节





--------------------------------------PS_VAE组-------------------------------------------

python /root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/sample_psvae_latent.py \
  --train_latent /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/latent/x_train.npy \
  --n_samples 1000 \
  --mode sample_only \
  --out_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/ps_vae/latent/psvae_init_latent.npy

CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode psvae \
  --pop_size 50 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version psvae_V_exp1_pop50



--------------------------------------LLM组---gpt-5.4-mini-------------------------------------------
Step1 生成smiles

#------------------------LLM model选择----------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#

#40轮迭代版本
python /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/generate_smiles_V1.py \
  --model gpt-5.4-mini \
  --task qm9_gap_min \
  --score_mode proxy_gap_min \
  --generations 40 \
  --n_candidates_per_gen 50 \
  --success_memory_size 3 \
  --failure_memory_size 3 \
  --target_total 200 \
  --temperature 0.5 \
  --max_tokens 2400
  
# Accepted total: 245/500 weak_llema_qm9_gap_min_gpt-5.4-mini_20260327_111612.smi


# 无多轮优化版本
cd /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/
python /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/generate_smiles_V1.py \
  --model gpt-5.4-mini \
  --task qm9_gap_min \
  --score_mode proxy_gap_min \
  --generations 1 \
  --n_candidates_per_gen 200 \
  --success_memory_size 0 \
  --failure_memory_size 0 \
  --target_total 200 \
  --temperature 0.5 \
  --max_tokens 2400
#  Accepted total: 187/200

#---------------------------#

Step2 提llm生成smiles的latent
python -u /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/extract_latent4llm.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt \
  --input_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260327_111612.smi \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent \
  --gpu 0

# 无多轮优化版本
python -u /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/extract_latent4llm.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt \
  --input_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260328_174203.smi \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent_0 \
  --gpu 0


#---------------------------#

Step3 进化生成latent

# 多轮优化版本
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode llm \
  --pop_size 50 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version llm_V_exp1_Opt_pop50

seed 42 123 2026

# 无多轮优化版本
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed_NoOpt.py \
  --init_mode llm \
  --pop_size 50 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version llm_V_exp1_NoOpt_pop50

--------------------------------------Hybrid组---------------------------------------------------------------
利用/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent/llm_init_latent.npy  作为 Hybrid组的seed latent

简单版本：不单独写 Hybrid 生成脚本

# 多轮优化版本

改/root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py中的LLM_LATENT_PATH 
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode hybrid \
  --hybrid_latent_path /tmp/not_exist.npy \
  --pop_size 50 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version hybrid_V_exp1_Opt_pop50 \
  --hybrid_keep_ratio 0.5\
  --hybrid_expand_ratio 0.3 \
  --hybrid_sigma 0.25
50_30_20


# 无多轮优化版本
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed_NoOpt.py \
  --init_mode hybrid \
  --hybrid_latent_path /tmp/not_exist.npy \
  --pop_size 50 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version hybrid_V_exp1_NoOpt_pop50 \
  --hybrid_keep_ratio 0.5\
  --hybrid_expand_ratio 0.3 \
  --hybrid_sigma 0.25
50_30_20

--------------------------------------汇总画图-------------------------------------------


python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/plot_3_gap_overlay.py \
  --psvae_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/psvae_psvae_V_exp1_pop50/summary.json \
  --llm_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_Opt_pop50/summary.json \
                  /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_NoOpt_pop50/summary.json \
  --hybrid_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_Opt_pop50/summary.json \
                     /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_NoOpt_pop50/summary.json \
  --out /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/Fig1_gap_overlay_30_50_20_All_pop50.png \
  --title "Average Predicted Gap Curves of Three Initialization Strategies" \
  --smooth_window 5

