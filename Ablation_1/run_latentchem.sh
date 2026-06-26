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
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version psvae_V_exp1


#Step3 latent解码为smiles❌
python /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/decode_lantent_qm9.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt \
  --z_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/psvae_psvae_v1/final_population_latent.npy \
  --out_csv /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/psvae_psvae_v1/final_population_decoded.csv \
  --gpu 0 \
  --max_atom_num 20 \
  --add_edge_th 0.5 \
  --temperature 0.6


--------------------------------------LLM组---gpt-5.4-mini-------------------------------------------
Step1 生成smiles

#------------------------LLM model选择----------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#

python /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/generate_smiles_V2-latentchem_copy.py \
  --task qm9_gap_min \
  --score_mode proxy_gap_min \
  --generations 1 \
  --n_candidates_per_gen 50 \
  --target_total 50 \
  --temperature 0.8 \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated_latentchem/

7/50 accepted
#---------------------------#
python /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/generate_smiles_V2-latentchem.py \
  --task qm9_gap_min \
  --score_mode proxy_gap_min \
  --generations 1 \
  --n_candidates_per_gen 50 \
  --target_total 50 \
  --temperature 0.8 \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated_latentchem/

  
#---------------------------#

Step2 提llm生成smiles的latent
python -u /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/extract_latent4llm.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_2/checkpoints/epoch=5-step=20076.ckpt \
  --input_path /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated/weak_llema_qm9_gap_min_gpt-5.4-mini_20260327_111612.smi \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent \
  --gpu 0
#---------------------------#

Step3 进化生成latent

# 多轮优化版本
python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode llm \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version llm_V_exp1_Opt

seed 42 123 2026

# 无多轮优化版本
python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode llm \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version llm_V_exp1_NoOpt

--------------------------------------Hybrid组---------------------------------------------------------------
利用/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent/llm_init_latent.npy  作为 Hybrid组的seed latent

简单版本：不单独写 Hybrid 生成脚本

# 多轮优化版本

改/root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py中的LLM_LATENT_PATH 
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode hybrid \
  --hybrid_latent_path /tmp/not_exist.npy \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version hybrid_V_exp1_Opt \
  --hybrid_keep_ratio 0.3\
  --hybrid_expand_ratio 0.5 \
  --hybrid_sigma 0.25



# 无多轮优化版本
CUDA_VISIBLE_DEVICES=2 python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/GeneticAlgorithm4Ablantion_1_fixed.py \
  --init_mode hybrid \
  --hybrid_latent_path /tmp/not_exist.npy \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --version hybrid_V_exp1_NoOpt \
  --hybrid_keep_ratio 0.3\
  --hybrid_expand_ratio 0.5 \
  --hybrid_sigma 0.25


--------------------------------------汇总画图-------------------------------------------

python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/plot_3_gap_overlay.py \
  --psvae_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/psvae_psvae_V_exp1/summary.json \
  --llm_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_NoOpt/summary.json \
  --hybrid_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_NoOpt/summary.json \
  --out /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/Fig1_gap_overlay_30_50_20_NoOpt.png \
  --title "Average Predicted Gap Curves of Three Initialization Strategies" \
  --smooth_window 5


python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/plot_3_gap_overlay.py \
  --psvae_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/psvae_psvae_V_exp1/summary.json \
  --llm_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_Opt/summary.json \
  --hybrid_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_Opt/summary.json \
  --out /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/Fig1_gap_overlay_30_50_20_Opt.png \
  --title "Average Predicted Gap Curves of Three Initialization Strategies" \
  --smooth_window 5

python /root/autodl-tmp/sweeteners_evolve/Ablation_1/hybrid/plot_3_gap_overlay.py \
  --psvae_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/psvae_psvae_V_exp1/summary.json \
  --llm_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_Opt/summary.json \
                  /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/llm_llm_V_exp1_NoOpt/summary.json \
  --hybrid_summaries /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_Opt/summary.json \
                     /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/hybrid_hybrid_V_exp1_NoOpt/summary.json \
  --out /root/autodl-tmp/sweeteners_evolve/Ablation_1/results/Fig1_gap_overlay_30_50_20_All.png \
  --title "Average Predicted Gap Curves of Three Initialization Strategies" \
  --smooth_window 5

