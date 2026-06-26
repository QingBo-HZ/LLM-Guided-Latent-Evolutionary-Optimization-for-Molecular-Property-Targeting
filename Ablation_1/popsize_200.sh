#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode psvae \
  --pop_size 200 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version psvae_V_exp1_pop200

--------------------------------------LLM组---gpt-5.4-mini-------------------------------------------
Step1 生成smiles

#------------------------LLM model选择----------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#

Step3 进化生成latent

# 多轮优化版本
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode llm \
  --pop_size 200 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version llm_V_exp1_Opt_pop200

seed 42 123 2026

# 无多轮优化版本
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed_NoOpt.py \
  --init_mode llm \
  --pop_size 200 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version llm_V_exp1_NoOpt_pop200

--------------------------------------Hybrid组---------------------------------------------------------------
利用/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent/llm_init_latent.npy  作为 Hybrid组的seed latent

简单版本：不单独写 Hybrid 生成脚本

# 多轮优化版本

改/root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py中的LLM_LATENT_PATH 
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode hybrid \
  --hybrid_latent_path /tmp/not_exist.npy \
  --pop_size 200 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version hybrid_V_exp1_Opt_pop200 \
  --hybrid_keep_ratio 0.3\
  --hybrid_expand_ratio 0.5 \
  --hybrid_sigma 0.25



# 无多轮优化版本
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed_NoOpt.py \
  --init_mode hybrid \
  --hybrid_latent_path /tmp/not_exist.npy \
  --pop_size 200 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version hybrid_V_exp1_NoOpt_pop200 \
  --hybrid_keep_ratio 0.3\
  --hybrid_expand_ratio 0.5 \
  --hybrid_sigma 0.25


--------------------------------------汇总画图-------------------------------------------


python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/plot_3_gap_overlay.py \
  --psvae_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/psvae_psvae_V_exp1_pop200/summary.json \
  --llm_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_Opt_pop200/summary.json \
                  /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_NoOpt_pop200/summary.json \
  --hybrid_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_Opt_pop200/summary.json \
                     /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_NoOpt_pop200/summary.json \
  --out /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/Fig1_gap_overlay_30_50_20_All_pop200.png \
  --title "Average Predicted Gap Curves of Three Initialization Strategies" \
  --smooth_window 5

