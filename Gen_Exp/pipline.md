conda activate molclr_pyg28

# 数据集划分
bash /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/scripts/split_data.sh

# 词典1000个子图

# 训练模型
bash train_chembl.sh

# 提取 ZINC latent + 计算 RDKit logP
python 01_encode_zinc_logp_latent.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt \
  --input /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_zinc/train/train.txt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train \
  --gpu 0

python 01_encode_zinc_logp_latent.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt \
  --input /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_zinc/valid/valid.txt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/valid \
  --gpu 1

python 01_encode_zinc_logp_latent.py \
  --ckpt /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt \
  --input /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_zinc/test/test.txt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/test \
  --gpu 2

# Step 2：训练 latent-logP predictor
python /root/autodl-tmp/sweeteners_evolve/Gen_Exp/02_train_logp_predictor.py \
  --train_latent /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train/zinc_logp_latent.npy \
  --train_meta /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train/zinc_logp_meta.csv \
  --valid_latent /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/valid/zinc_logp_latent.npy \
  --valid_meta /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/valid/zinc_logp_meta.csv \
  --test_latent /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/test/zinc_logp_latent.npy \
  --test_meta /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/test/zinc_logp_meta.csv \
  --out_dir /root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/logp_predictor \
  --gpu 0 \
  --hidden_dim 512 \
  --dropout 0.05 \
  --epochs 400 \
  --batch_size 512 \
  --lr 8e-4 \
  --weight_decay 1e-6 \
  --patience 50





# 分开计算logD
conda activate llema
cd /root/autodl-tmp/sweeteners_evolve/Ablation_2/LLM/RTlogD-main

python batch_predict_logd_chembl.py \
  --train_smiles /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_chembl/train/train.txt \
  --val_smiles /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_chembl/valid/valid.txt \
  --test_smiles /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_chembl/test/test.txt \
  --out_dir /root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_chembl/logd_labels \
  --device cpu





