主实验流程
conda activate molclr_pyg28

1. random search
CUDA_VISIBLE_DEVICES=1 python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/1_random_search/random_search.py \
  --n_samples 100000 \
  --batch_size 100 \
  --success_threshold 0.15 \
  --topk_per_step 10 \
  --version random_search_V2

2. smiles_ga
CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA/smiles_ga_qm9_childselect.py \
  --train_smiles_csv /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/qm9_ext_pred/labeled_split/train_labeled.csv \
  --smiles_col smiles \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 30 \
  --mut_prob 0.20 \
  --cross_prob 0.20 \
  --fragment_lib_max_mols 50000 \
  --success_threshold 0.15 \
  --warm_start \
  --warm_start_frac 0.8 \
  --warm_start_gap_upper 0.25 \
  --child_trials 10 \
  --tourn_size 5 \
  --random_immigrant_frac 0.05 \
  --seed 42 \
  --version smiles_childselect_v1


3. latent_ga_llm

CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/3_latent_GA_noLLM/latent_GA_random.py \
  --init_mode psvae \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --success_threshold 0.15 \
  --version train_random_V2
4.
CUDA_VISIBLE_DEVICES=1 python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/4_latent_GA_LLM/latent_GA_LLM.py \
  --init_mode llm \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --success_threshold 0.15 \
  --version llm_V2

5.
CUDA_VISIBLE_DEVICES=1 python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/5_Ours/latent_ours.py \
  --init_mode llm \
  --pop_size 100 \
  --n_gen 1000 \
  --elite_size 20 \
  --cross_prob 0.8 \
  --mut_prob 0.08 \
  --mut_eta 20 \
  --patience 20 \
  --seed 42 \
  --success_threshold 0.15 \
  --version ours_V2

# 分子演化图
# 1
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/draw_smiles_grid_svg.py \
  --csv_path /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/1_random_search/random_search_random_search_V2/evolution_path_full.csv \
  --mode best_path \
  --n_pick 5 \
  --out_svg /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/1_random_search/fig2e_5mols.svg
# 2
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/draw_smiles_grid_svg.py \
  --csv_path /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA/fragment_ga_smiles_childselect_v1/evolution_path_full.csv \
  --mode best_path \
  --n_pick 5 \
  --out_svg /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA/fig2e_5mols.svg
# 3
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/draw_smiles_grid_svg.py \
  --csv_path /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/3_latent_GA_noLLM/psvae_train_random_V2/evolution_path.csv \
  --mode best_path \
  --n_pick 5 \
  --out_svg /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/3_latent_GA_noLLM/figure4_5mols.svg
# 4
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/draw_smiles_grid_svg.py \
  --csv_path /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/4_latent_GA_LLM/llm_llm_V2/evolution_path.csv \
  --mode best_path \
  --n_pick 5 \
  --out_svg /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/4_latent_GA_LLM/figure4_5mols.svg
# 5
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/draw_smiles_grid_svg.py \
  --csv_path /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/5_Ours/llm_ours_V2/evolution_path.csv \
  --mode best_path \
  --n_pick 5 \
  --out_svg /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/5_Ours/figure4_5mols.svg

# eval计算
# 1
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/calc_sr_and_eval.py \
  --final_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/1_random_search/random_search_random_search_V1/final_population_random.csv \
  --progress_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/1_random_search/random_search_random_search_V1/progress_metrics.csv \
  --threshold 0.15
# 2
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/calc_sr_and_eval.py \
  --final_gap_col "gap" \
  --final_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA/fragment_ga_main_v1/final_population_fragment_ga.csv \
  --progress_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/2_smiles_GA/fragment_ga_main_v1/progress_metrics.csv \
  --threshold 0.15
# 3
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/calc_sr_and_eval.py \
  --final_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/3_latent_GA_noLLM/train_random_train_random_V1/final_population_train_random_V1.csv \
  --progress_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/3_latent_GA_noLLM/train_random_train_random_V1/progress_metrics.csv \
  --threshold 0.15
# 4
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/calc_sr_and_eval.py \
  --final_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/4_latent_GA_LLM/llm_llm_V1/final_population_llm_V1.csv \
  --progress_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/4_latent_GA_LLM/llm_llm_V1/progress_metrics.csv \
  --threshold 0.15
# 5
python /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/calc_sr_and_eval.py \
  --final_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/5_Ours/llm_ours_V1/final_population_ours_V1.csv \
  --progress_csv /root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/5_Ours/llm_ours_V1/progress_metrics.csv \
  --threshold 0.15